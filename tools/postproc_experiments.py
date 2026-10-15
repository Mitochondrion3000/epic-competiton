# Дешёвые проверки «что даёт знание об ошибках» на готовых предсказаниях (CPU, без обучения сети).
# run(z, gt, ok, fold, base, prev, out): z — стандартизованный лог-скор, fold — 0/1/-1 (на какой половине позиция),
# base/prev — коды нуклеотида в позиции и предыдущего в направлении цепи (0..3 ACGT, 4+ иначе).
# Схема: параметры и alpha выбираются на одной половине, качество измеряется на другой (обе направления), метрики официальные.
import json, itertools, numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/../simple_model")
from scoring import official_score, rank_power
ALPHAS = [0.6, 0.75, 0.9]

def run(z, gt, ok, fold, base, prev, out, cand_frac=0.03):
    z = z.astype(np.float32); rows = []; idx = {f: np.flatnonzero((fold == f) & ok) for f in (0, 1)}
    def measure(name, params, s):
        for a in ALPHAS:
            for f in (0, 1):
                r = rank_power(s[idx[f]], a); m = official_score(r, gt[idx[f]])
                rows.append(dict(variant=name, params=params, alpha=a, fold=f, **m))
    measure("baseline", "", z)
    for sg in (1, 2, 3): measure("gauss", f"sigma={sg}", gaussian_filter1d(z, sg).astype(np.float32))
    for k, d in itertools.product((2, 5), (0.25, 0.5, 1.0)):
        mx = maximum_filter1d(z, 2 * k + 1); measure("nms", f"k={k},delta={d}", z - d * (z < mx).astype(np.float32))
    for k, w in itertools.product((2, 5, 10), (0.25, 0.5, 1.0)):
        measure("neighbor_max", f"k={k},w={w}", z + w * maximum_filter1d(z, 2 * k + 1))
    pyr = ((prev == 1) | (prev == 3)) & (base == 0)           # пара CA/TA в позиции старта
    for c in (0.25, 0.5, 1.0): measure("ca_penalty", f"c={c}", z - c * pyr.astype(np.float32))
    # --- reranker (LightGBM) на кандидатах: верхние cand_frac по z
    import lightgbm as lgb
    thr = np.quantile(z[ok], 1 - cand_frac); cand = ok & (z >= thr)
    ind = (z >= thr).astype(np.float32)
    def feats(use_seq, use_ctx):
        F = [z]
        if use_ctx:
            for k in (1, 2, 5, 10, 50): F += [maximum_filter1d(z, 2 * k + 1) - z]
            F += [uniform_filter1d(z, 101) - z, uniform_filter1d(ind, 101), uniform_filter1d(ind, 21)]
        if use_seq: F += [base.astype(np.float32), prev.astype(np.float32), pyr.astype(np.float32)]
        return np.stack(F, 1)
    for name, us, uc in (("rerank_ctx", False, True), ("rerank_seq", True, False), ("rerank_ctx_seq", True, True)):
        X_all = feats(us, uc); P = {}
        for tr in (0, 1):
            itr = np.flatnonzero(cand & (fold == tr)); m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            m.fit(X_all[itr], (gt[itr] > 0).astype(int)); P[tr] = m
        for tr in (0, 1):
            ev = 1 - tr; s = z.copy(); ic = np.flatnonzero(cand & (fold == ev))
            s[ic] = z.max() + 1 + P[tr].predict_proba(X_all[ic])[:, 1]
            for a in ALPHAS:
                r = rank_power(s[idx[ev]], a); rows.append(dict(variant=name, params=f"trained_on_fold{tr}", alpha=a, fold=ev, **official_score(r, gt[idx[ev]])))
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out + ".grid.csv", index=False)
    # выбор параметров и alpha на «tune»-половине, оценка на другой
    res = []
    for v, g in df.groupby("variant"):
        for ev in (0, 1):
            tr = 1 - ev
            if v.startswith("rerank"):
                best = g[(g.fold == ev)].sort_values("AUPRC", ascending=False)  # reranker уже обучен на другой половине; alpha выбираем на tr-половине baseline-подобно: берём alpha лучшего baseline
                a = df[(df.variant == "baseline") & (df.fold == tr)].sort_values("AUPRC", ascending=False).iloc[0].alpha
                pick = g[(g.fold == ev) & (g.alpha == a)].iloc[0]
            else:
                gt_ = g[g.fold == tr].sort_values("AUPRC", ascending=False).iloc[0]
                pick = g[(g.fold == ev) & (g.params == gt_.params) & (g.alpha == gt_.alpha)].iloc[0]
            res.append(dict(variant=v, params=pick.params, alpha=pick.alpha, eval_fold=ev, AUPRC=pick.AUPRC, Spearman=pick.Spearman))
    R = pd.DataFrame(res); R.to_csv(out + ".summary.csv", index=False)
    S = R.groupby("variant")[["AUPRC", "Spearman"]].mean(); b = S.loc["baseline"]
    S["dAUPRC_%"] = (S.AUPRC / b.AUPRC - 1) * 100; S["dSpearman"] = S.Spearman - b.Spearman
    S.to_csv(out + ".table.csv"); print(S.round(4).to_string()); print(R.to_string())
    return S

def seq_codes(chrom, start, end, strand, genome, total):
    """Коды основания в позиции и предыдущего (в направлении цепи) для всех позиций, склеенных по строкам. strand: 0 (+) / 1 (-)."""
    COMP = np.array([3, 2, 1, 0, 4, 5, 6, 7, 8]); base = np.empty(total, np.uint8); prev = np.empty(total, np.uint8); o = 0
    for c, s, e, st in zip(chrom, start, end, strand):
        g = genome[c]; ext = np.pad(g[max(s - 1, 0):e + 1], (1 if s == 0 else 0, max(e + 1 - len(g), 0)), constant_values=8); n = e - s
        b, p_ = (COMP[ext[1:n + 1]], COMP[ext[2:n + 2]]) if st else (ext[1:n + 1], ext[0:n])
        base[o:o + n] = b % 4 if False else np.where(b < 8, b % 4, 4); prev[o:o + n] = np.where(p_ < 8, p_ % 4, 4); o += n
    return base, prev
