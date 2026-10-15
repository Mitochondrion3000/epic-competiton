# Общий расчёт статистик ошибок для отчёта (сохраняет компактный JSON для графиков).
# compute(p, gt, ok, loc, genome, out): p — предсказания, gt — счёт стартов, ok — маска скоринга (не N),
# loc(idx) -> (контиги, позиции, цепь 0/1) для индексов вектора, genome — {контиг: коды 0..8}.
import json, numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.stats import mannwhitneyu
COMP = np.array([3, 2, 1, 0, 4, 5, 6, 7, 8])
KS = [0, 1, 2, 5, 10, 20]
BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 49, "10-49"), (50, 10**9, "50+")]

def window(genome, c, g, st, h):
    s = genome[c]; a = s[max(g - h, 0):g + h + 1]
    a = np.pad(a, (max(h - g, 0), max(g + h + 1 - len(s), 0)), constant_values=8)
    return COMP[a[::-1]] if st else a

def compute(p, gt, ok, loc, genome, out, cap=20000, seed=0):
    rng = np.random.default_rng(seed); R = {}
    pos = (gt > 0) & ok; N = int(pos.sum()); R["meta"] = dict(N_starts=N, N_scored=int(ok.sum()), n_gt1=int((pos & (gt == 1)).sum()))
    thr = np.partition(p[ok], -N)[-N]; top = (p >= thr) & ok; R["meta"]["thr"] = float(thr)
    # --- допуск ±k: найден ли старт; точность топ-N
    tol = []
    for k in KS:
        nearp = maximum_filter1d(pos.astype(np.uint8), 2 * k + 1) > 0 if k else pos
        neart = maximum_filter1d(top.astype(np.uint8), 2 * k + 1) > 0 if k else top
        tol.append(dict(k=k, recall=float(neart[pos].mean()), precision=float(nearp[top].mean())))
    R["tolerance"] = tol
    found = {k: (maximum_filter1d(top.astype(np.uint8), 2 * k + 1) > 0 if k else top) for k in (0, 5)}
    # --- доля пропусков по силе старта
    R["strength"] = [dict(label=l, n=int((pos & (gt >= a) & (gt <= b)).sum()),
                          fn_k0=float(1 - found[0][pos & (gt >= a) & (gt <= b)].mean()),
                          fn_k5=float(1 - found[5][pos & (gt >= a) & (gt <= b)].mean())) for a, b, l in BINS]
    # --- одиночные / кластерные старты
    cs = np.r_[0, np.cumsum(pos.astype(np.int32))]; idxp = np.flatnonzero(pos); L = len(p)
    near = cs[np.minimum(idxp + 51, L)] - cs[np.maximum(idxp - 50, 0)] - 1
    cl = []
    for a, b, l in [(0, 0, "0 (одиночный)"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 10**9, "11+")]:
        m = (near >= a) & (near <= b)
        cl.append(dict(label=l, n=int(m.sum()), share=float(m.mean()), fn_k0=float(1 - found[0][idxp[m]].mean()), fn_k5=float(1 - found[5][idxp[m]].mean())))
    R["cluster"] = cl
    # --- PR (строгая): все позитивы + 2 млн негативов с весами
    neg = np.flatnonzero(ok & ~pos); sub = rng.choice(neg, min(2_000_000, len(neg)), replace=False); w_neg = len(neg) / len(sub)
    ii = np.r_[idxp, sub]; w = np.r_[np.ones(len(idxp)), np.full(len(sub), w_neg)]; y = np.r_[np.ones(len(idxp)), np.zeros(len(sub))]
    o = np.argsort(-p[ii], kind="stable"); tp = np.cumsum((w * y)[o]); fp = np.cumsum((w * (1 - y))[o])
    prec, rec = tp / (tp + fp), tp / N; sel = np.unique(np.linspace(0, len(o) - 1, 400).astype(int))
    R["pr"] = dict(recall=rec[sel].tolist(), precision=prec[sel].tolist(), auprc=float(np.sum(np.diff(np.r_[0, rec]) * prec)), baseline=N / int(ok.sum()))
    # --- распределение предсказаний (перцентиль среди scored) по силе старта
    ref = np.sort(rng.choice(p[ok], min(1_000_000, int(ok.sum())), replace=False)); pct = lambda x: np.searchsorted(ref, x) / len(ref)
    qs = []
    for a, b, l in [(0, 0, "0 (нет)")] + BINS:
        m = ok & (gt >= a) & (gt <= b)
        ix = np.flatnonzero(m); ix = rng.choice(ix, min(len(ix), 200000), replace=False)
        qs.append(dict(label=l, q=np.quantile(pct(p[ix]), [.05, .25, .5, .75, .95]).tolist()))
    R["pred_by_gt"] = qs
    h2 = np.zeros((7, 20)); rows_ = [(0, 0)] + [(a, b) for a, b, _ in BINS]
    for r, (a, b) in enumerate(rows_):
        m = ok & (gt >= a) & (gt <= b); ix = np.flatnonzero(m); ix = rng.choice(ix, min(len(ix), 200000), replace=False)
        h2[r] = np.histogram(pct(p[ix]), bins=20, range=(0, 1))[0] / len(ix)
    R["hist2d"] = dict(rows=["0 (нет)"] + [l for _, _, l in BINS], cols=20, z=h2.tolist())
    # --- группы для признаков
    grp = np.full(L, "", object); grp[pos & top] = "TP"; grp[pos & ~top] = "FN"; grp[~pos & top] = "FP"
    for g in ("TP", "FN", "FP"):
        ix = np.flatnonzero(grp == g)
        if len(ix) > cap: grp[rng.choice(ix, len(ix) - cap, replace=False)] = ""
    grp[rng.choice(np.flatnonzero(ok & ~top & ~pos), cap, replace=False)] = "BG"
    idx = np.flatnonzero(grp != ""); ch, gp, st = loc(idx)
    recs = []; nuc = {g: np.zeros((21, 4)) for g in ("TP", "FN", "FP", "BG")}; nn = {g: 0 for g in nuc}
    for c, g, s_, i in zip(ch, gp, st, idx):
        w_ = window(genome, c, int(g), int(s_), 100); u = w_ % 4; o_ = w_ != 8; low = (w_ >= 4) & (w_ < 8)
        cc, gg = np.sum((u == 1) & o_), np.sum((u == 2) & o_); cg = np.sum((u[:-1] == 1) & (u[1:] == 2) & o_[:-1] & o_[1:])
        k3 = u[:-2] * 16 + u[1:-1] * 4 + u[2:]; cnt = np.bincount(k3[o_[:-2]], minlength=64) + 0.0; q = cnt / max(cnt.sum(), 1)
        recs.append((grp[i], (cc + gg) / max(o_.sum(), 1), cg / max(cc * gg / max(o_.sum(), 1), 1e-9), low.mean(), -(q[q > 0] * np.log2(q[q > 0])).sum(),
                     int(cs[min(i + 51, L)] - cs[max(i - 50, 0)] - (1 if pos[i] else 0))))
        c10 = w_[90:111]
        if (c10 < 4).all(): nuc[grp[i]][np.arange(21), c10] += 1; nn[grp[i]] += 1
    import pandas as pd
    df = pd.DataFrame(recs, columns=["grp", "gc", "cpg_oe", "lower", "ent3", "n_starts_pm50"])
    feats = {}
    rngs = dict(gc=(0, 1), cpg_oe=(0, 2.5), lower=(0, 1), ent3=(3, 6), n_starts_pm50=(0, 40))
    for f, (lo, hi) in rngs.items():
        d = dict(range=[lo, hi], hist={}, median={}, n={})
        for g in ("TP", "FN", "FP", "BG"):
            x = df[df.grp == g][f].dropna().values; d["hist"][g] = (np.histogram(np.clip(x, lo, hi), bins=30, range=(lo, hi))[0] / max(len(x), 1)).tolist()
            d["median"][g] = float(np.median(x)); d["n"][g] = int(len(x))
        for a_, b_ in (("FN", "TP"), ("FP", "BG"), ("FP", "TP")):
            xa, xb = df[df.grp == a_][f].dropna().values, df[df.grp == b_][f].dropna().values
            u = mannwhitneyu(xa, xb); d[f"{a_}_vs_{b_}"] = dict(p=float(u.pvalue), auc=float(u.statistic / (len(xa) * len(xb))))
        feats[f] = d
    R["features"] = feats; R["group_n"] = {g: int((df.grp == g).sum()) for g in nuc}
    R["nuc"] = {g: (nuc[g] / max(nn[g], 1)).tolist() for g in nuc}; R["nuc_n"] = nn
    # динуклеотид (предыдущая + позиция старта) из окна ±1
    di = {g: {} for g in nuc}
    for c, g, s_, i in zip(ch, gp, st, idx):
        w_ = window(genome, c, int(g), int(s_), 1); k = "ACGTN"[w_[0] % 4 if w_[0] < 8 else 4] + "ACGTN"[w_[1] % 4 if w_[1] < 8 else 4]
        di[grp[i]][k] = di[grp[i]].get(k, 0) + 1
    R["dinuc"] = {g: {k: v / max(sum(d.values()), 1) for k, v in d.items()} for g, d in di.items()}
    json.dump(R, open(out, "w"), ensure_ascii=False)
    return R
