# Предсказание силы старта на верхних позициях: LightGBM-регрессия log(count) на кандидатах (верхние cand_frac по z).
# run(z, gt, ok, fold, base, prev, out, z3=None): те же входы, что у postproc_experiments.run (gt — счета, fold 0/1/-1).
# Схема: обучение на одной половине, оценка на другой (обе стороны); alpha выбирается по baseline на обучающей половине.
# Монотонность по z (и z3) защищает AUPRC; позиции вне кандидатов не меняются.
import sys, os, numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
sys.path.insert(0, os.path.dirname(__file__) + "/../simple_model")
from scoring import official_score, rank_power
ALPHAS = [0.6, 0.75, 0.9]

def feats(z, base, prev, z3=None):
    cg = ((prev == 1) & (base == 2)).astype(np.float32)                 # CpG в позиции старта (в направлении цепи)
    gc = ((base == 1) | (base == 2)).astype(np.float32)
    pyr = (((prev == 1) | (prev == 3)) & (base == 0)).astype(np.float32)  # CA/TA
    F, names = [z], ["z"]
    if z3 is not None: F.append(z3); names.append("z3")
    for k in (2, 5, 10, 50):
        F.append(maximum_filter1d(z, 2 * k + 1) - z); names.append(f"dmax{k}")
    for k in (5, 50):
        F.append(uniform_filter1d(z, 2 * k + 1) * (2 * k + 1)); names.append(f"sum{k}")
    hi = (z >= np.quantile(z, 0.97)).astype(np.float32)
    F.append(uniform_filter1d(hi, 101)); names.append("dens50")
    F += [uniform_filter1d(gc, 201), uniform_filter1d(cg, 201), pyr, base.astype(np.float32)]; names += ["gc100", "cpg100", "CA_TA", "base"]
    return np.stack(F, 1).astype(np.float32), names

def run(z, gt, ok, fold, base, prev, out, z3=None, fracs=(0.03, 0.05)):
    import lightgbm as lgb, pandas as pd
    z = z.astype(np.float32); X, names = feats(z, base, prev, z3)
    mono = [1 if n in ("z", "z3") else 0 for n in names]
    idx = {f: np.flatnonzero((fold == f) & ok) for f in (0, 1)}; rows = []
    for a in ALPHAS:
        for f in (0, 1): rows.append(dict(variant="baseline", frac=0, alpha=a, fold=f, **official_score(rank_power(z[idx[f]], a), gt[idx[f]])))
    for frac in fracs:
        thr = np.quantile(z[ok], 1 - frac); cand = ok & (z >= thr)
        for tr in (0, 1):
            ev = 1 - tr; itr = np.flatnonzero(cand & (fold == tr) & (gt > 0)); ic = np.flatnonzero(cand & (fold == ev))
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, monotone_constraints=mono, verbose=-1)
            m.fit(X[itr], np.log(gt[itr])); p = m.predict(X[ic]); pz = (p - p.mean()) / (p.std() + 1e-9)
            zz = (z[ic] - z[ic].mean()) / (z[ic].std() + 1e-9)
            for name, v in (("str_only", p), ("str+z", pz + zz), ("str+0.5z", pz + 0.5 * zz)):
                s = z.copy(); s[ic] = z.max() + 1 + (v - v.min()) / (np.ptp(v) + 1e-9)
                for a in ALPHAS: rows.append(dict(variant=f"{name}@{frac}", frac=frac, alpha=a, fold=ev, **official_score(rank_power(s[idx[ev]], a), gt[idx[ev]])))
    df = pd.DataFrame(rows); df.to_csv(out + ".grid.csv", index=False); res = []
    for v, g in df.groupby("variant"):
        for ev in (0, 1):
            a = df[(df.variant == "baseline") & (df.fold == 1 - ev)].sort_values("AUPRC", ascending=False).iloc[0].alpha
            pk = g[(g.fold == ev) & (g.alpha == a)].iloc[0]; res.append(dict(variant=v, alpha=a, eval_fold=ev, AUPRC=pk.AUPRC, Spearman=pk.Spearman))
    R = pd.DataFrame(res); S = R.groupby("variant")[["AUPRC", "Spearman"]].mean(); b = S.loc["baseline"]
    S["dAUPRC_%"] = (S.AUPRC / b.AUPRC - 1) * 100; S["dSpearman"] = S.Spearman - b.Spearman
    S.to_csv(out + ".table.csv"); print(S.round(4).to_string()); return S
