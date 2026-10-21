# CPU: реранкер «скор сети + окружение» на честных скорах v3 для report моли (4 Мб окон, 1953 окна).
# Позиции окон восстанавливаем так же, как make_eval_set в epic_v3_bpnet.ipynb (seed 0), и сверяем метки с сохранёнными.
# Кандидаты — верхние CAND по скору сети; LightGBM учится на одной половине (блоки по 1 Мб), оценка на другой, и наоборот.
import glob, json, os, subprocess, sys, time, zlib
import numpy as np, pandas as pd
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=False)
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.ndimage import maximum_filter1d, uniform_filter1d
SP = "ilPloInte3.2"; W, C = 4096, 1024; STEP = W - 2 * C; EVAL_BP = 4_000_000
MASK = int(os.environ.get("EPIC_MASK", "60")); WIN = (300, 1000, 3000); CAND = float(os.environ.get("EPIC_CAND", "0.03")); PAD = MASK + max(WIN) + 50
t0 = time.time(); rng = np.random.default_rng(0)
cd = os.path.dirname(sorted(glob.glob(f"/kaggle/input/**/cache/{SP}/DONE", recursive=True))[0])
m = pd.read_pickle(f"{cd}/meta.pkl"); flat = np.load(f"{cd}/genome.npy", mmap_mode="r")
bounds = dict(zip(m["names"], zip(m["offs"][:-1], m["offs"][1:])))
gp = [p for p in glob.glob("/kaggle/input/**/holdout_groups.json", recursive=True) if "pred-moth" in p][0]
report = json.load(open(gp))["report"]; print("report-контигов", len(report), "файл групп", gp, flush=True)
sc = [p for p in glob.glob("/kaggle/input/**/report_scores_ilPloInte3.2.npz", recursive=True) if "pred-moth" in p][0]
S = np.load(sc); y_saved, ctrl, fp = S["y"], S["ctrl"].astype(np.float32), S["fp"].astype(np.float32); print("позиций", len(y_saved), flush=True)
raw = lambda c: np.asarray(flat[bounds[c][0]:bounds[c][1]])
wl = {c: g[["start", "end"]].values for c, g in m["wl"].groupby("chrom")}
inwl = {}
for c in report:
    a = np.zeros(bounds[c][1] - bounds[c][0], bool)
    for s, e in wl.get(c, []): a[s:e] = True
    inwl[c] = a
starts = [(c, s) for c in report for s in range(-C, len(inwl[c]) - C, STEP) if inwl[c][max(s + C, 0):s + C + STEP].any()]
starts = [starts[i] for i in rng.permutation(len(starts))[: max(1, EVAL_BP // STEP)]]; n = len(starts); print("окон", n, flush=True)
M = np.zeros((n, STEP), bool); Y = np.zeros((n, 2, STEP), np.float32)
gcache = {}
def G(c):
    if c not in gcache: gcache[c] = raw(c)
    return gcache[c]
for k, (c, s) in enumerate(starts):
    g = G(c); L = len(g); lo, hi = max(s + C, 0), min(s + C + STEP, L)
    if hi > lo: M[k, lo - s - C:hi - s - C] = inwl[c][lo:hi] & (g[lo:hi] != 8)
    t = m["track"].get(c)
    if t is not None:
        i0, i1 = np.searchsorted(t["pos"], [s + C, s + C + STEP])
        Y[k, t["strand"][i0:i1], t["pos"][i0:i1] - s - C] = t["tot"][i0:i1]
MM = np.broadcast_to(M[:, None, :], Y.shape); y_rec = Y[MM]
assert len(y_rec) == len(y_saved) and np.allclose(y_rec, y_saved), "восстановленные метки не совпали с сохранёнными: позиции окон неверны"
print("метки восстановлены и совпали с сохранёнными", flush=True)
win, strand, off = np.nonzero(MM)
s_all = (ctrl + fp) / 2; y = y_saved
fold_w = np.array([zlib.crc32(f"{c}:{(s + C) // 1_000_000}".encode()) % 2 for c, s in starts]); fold = fold_w[win]
thr = np.quantile(s_all, 1 - CAND); cand = s_all >= thr; print("кандидатов", int(cand.sum()), "положительных среди них", int((y[cand] > 0).sum()), "из", int((y > 0).sum()), flush=True)
# признаки скоров внутри окна
D = np.full((n, 2, STEP), -1e9, np.float32); D[MM] = s_all
F = {"s": s_all, "ctrl": ctrl, "fp": fp, "d_models": ctrl - fp}
mx = np.full(len(s_all), 0, np.float32)
for kk in (1, 5, 10, 50):
    f = maximum_filter1d(D, 2 * kk + 1, axis=2, mode="constant", cval=-1e9)[MM]; F[f"nmax{kk}"] = s_all - f
Dm = np.where(D < -1e8, np.nan, D); Dm0 = np.where(np.isnan(Dm), np.nanmean(Dm), Dm)
F["wmean101"] = s_all - uniform_filter1d(Dm0, 101, axis=2)[MM]
# признаки генома только для кандидатов
ci = np.flatnonzero(cand); cw, cs, co = win[ci], strand[ci], off[ci]; GEN = {}
names = [f"{d}{w}_{t}" for d in ("up", "dn") for w in WIN for t in ("gc", "cpg", "a", "t")] + ["cm1", "c0", "cp1"]
Gm = np.full((len(ci), len(names)), np.nan, np.float32)
for k in np.unique(cw):
    c, s = starts[k]; g = G(c); L = len(g); lo = s + C - PAD; hi = s + C + STEP + PAD
    seg = np.full(hi - lo, 4, np.uint8); a, b = max(lo, 0), min(hi, L)
    seg[a - lo:b - lo] = np.where(g[a:b] < 8, g[a:b] % 4, 4)
    for st in (0, 1):
        sel = np.flatnonzero((cw == k) & (cs == st))
        if len(sel) == 0: continue
        o = seg if st == 0 else np.where(seg < 4, 3 - seg, seg)[::-1]; Lw = len(o)
        idx = (PAD + co[sel]) if st == 0 else (Lw - 1 - (PAD + co[sel]))
        PG = np.r_[0, np.cumsum((o == 1) | (o == 2), dtype=np.int32)]; PA = np.r_[0, np.cumsum(o == 0, dtype=np.int32)]
        PT = np.r_[0, np.cumsum(o == 3, dtype=np.int32)]; PC = np.r_[0, np.cumsum(o == 1, dtype=np.int32)]
        PCG = np.r_[0, np.cumsum((o[:-1] == 1) & (o[1:] == 2), dtype=np.int32), 0]
        col = 0
        for d in ("up", "dn"):
            for w in WIN:
                lo_, hi_ = (idx - MASK - w, idx - MASK) if d == "up" else (idx + MASK + 1, idx + MASK + 1 + w)
                lo_ = np.clip(lo_, 0, Lw); hi_ = np.clip(hi_, 0, Lw); nn = np.maximum(hi_ - lo_, 1)
                cc = PC[hi_] - PC[lo_]; gg = (PG[hi_] - PG[lo_]) - cc; cg = PCG[np.maximum(hi_ - 1, lo_)] - PCG[lo_]
                Gm[sel, col] = (PG[hi_] - PG[lo_]) / nn; Gm[sel, col + 1] = cg * nn / np.maximum(cc * gg, 1)
                Gm[sel, col + 2] = (PA[hi_] - PA[lo_]) / nn; Gm[sel, col + 3] = (PT[hi_] - PT[lo_]) / nn; col += 4
        Gm[sel, col] = o[np.clip(idx - 1, 0, Lw - 1)]; Gm[sel, col + 1] = o[idx]; Gm[sel, col + 2] = o[np.clip(idx + 1, 0, Lw - 1)]
print(f"признаки готовы {(time.time() - t0) / 60:.1f} мин", flush=True)
Xs = np.stack([F[k][ci] for k in F], 1); Xg = Gm; names_s = list(F)
sets = {"скор+окно": (Xs, names_s), "скор+окно+геном": (np.hstack([Xs, Xg]), names_s + names)}
def metr(score, yy):
    ap = average_precision_score(yy > 0, score); pos = yy > 0
    r = lambda v: np.unique(v, return_inverse=True)[1].astype(np.float64)
    sp = np.corrcoef(r(score[pos]), r(yy[pos]))[0, 1]
    return ap, sp
rows = []
for name, (X, nm_) in sets.items():
    P = np.zeros(len(ci))
    for tr in (0, 1):
        itr = fold[ci] == tr; ite = ~itr
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        mdl.fit(X[itr], (y[ci][itr] > 0).astype(int)); P[ite] = mdl.predict_proba(X[ite])[:, 1]
        if name == "скор+окно+геном" and tr == 0:
            imp = pd.Series(mdl.feature_importances_, nm_).sort_values(ascending=False); print(imp.head(10).to_string(), flush=True)
    for variant in ("замена", "сумма w=0.5", "сумма w=1"):
        sc_new = s_all.copy()
        lg = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
        if variant == "замена": sc_new[ci] = s_all.max() + 1 + P
        else: sc_new[ci] = s_all[ci] + float(variant.split("=")[1]) * lg
        for f in (0, 1):
            e = fold == f; b = metr(s_all[e], y[e]); v = metr(sc_new[e], y[e])
            rows.append(dict(набор=name, вариант=variant, fold=f, AUPRC_база=b[0], AUPRC=v[0], dAUPRC_pct=(v[0] / b[0] - 1) * 100, Sp_база=b[1], Sp=v[1], dSp=v[1] - b[1]))
R = pd.DataFrame(rows); pd.set_option("display.width", 200); print(R.round(4).to_string(), flush=True)
S2 = R.groupby(["набор", "вариант"])[["dAUPRC_pct", "dSp"]].mean(); print(S2.round(4).to_string(), flush=True)
R.to_csv("/kaggle/working/rerank_result.csv", index=False); print("готово", f"{(time.time() - t0) / 60:.1f} мин")
