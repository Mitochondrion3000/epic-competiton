# CPU-проверка идеи «модель окружения»: центр ±MASK нт замаскирован, по сторонам считаем GC / CpG o/e / A / T в окнах разной ширины
# (отдельно выше и ниже по цепи). Кандидаты — позиции с правилом Инициатора (предыдущий C/T, сам A/G), без предсказаний нашей сети.
# Метка — есть ли в позиции наблюдаемый старт (tot>0). Деление по контигам. Сравниваем три набора признаков: центр, окружение, оба.
import glob, os, time, numpy as np, pandas as pd, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=False)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
SP = os.environ.get("EPIC_SPECIES", "ilPloInte3.2"); MASK = int(os.environ.get("EPIC_MASK", "25")); RATE = float(os.environ.get("EPIC_RATE", "0.1"))
WIN = tuple(int(x) for x in os.environ.get("EPIC_WIN", "30,100,300,1000,3000").split(",")); t0 = time.time(); rng = np.random.default_rng(0)
d = os.path.dirname(sorted(glob.glob(f"/kaggle/input/**/cache/{SP}/DONE", recursive=True))[0])
m = pd.read_pickle(f"{d}/meta.pkl"); flat = np.load(f"{d}/genome.npy", mmap_mode="r")
print(SP, "контигов", len(m["names"]), "в whitelist", m["wl"].chrom.nunique(), flush=True)
wl = {c: g[["start", "end"]].values for c, g in m["wl"].groupby("chrom")}
bounds = dict(zip(m["names"], zip(m["offs"][:-1], m["offs"][1:])))
order = [c for c in m["train"]]; rng.shuffle(order)
tot_bp = sum(bounds[c][1] - bounds[c][0] for c in order); cum = 0; part = {}
for c in order: part[c] = int(cum >= tot_bp / 2); cum += bounds[c][1] - bounds[c][0]   # 0 = обучение, 1 = проверка

def cs(x): return np.concatenate([[0], np.cumsum(x, dtype=np.int32)])
def side(P, lo, hi):            # доля/число по окну [lo, hi) (в ориентированных координатах), P — префиксные суммы
    return P[hi] - P[lo]
rows = []
for c in order:
    a, b = bounds[c]; g = np.asarray(flat[a:b]); g = np.where(g < 8, g % 4, 4).astype(np.uint8); L = len(g)   # строчные (маскированные повторы) -> те же основания, N -> 4
    if L < 2 * (MASK + 50): continue
    ok = np.zeros(L, bool)
    for s, e in wl.get(c, []): ok[s:e] = True
    tr = m["track"].get(c)
    for st in (0, 1):
        s = g if st == 0 else np.where(g < 4, 3 - g, g)[::-1]        # ориентированная последовательность (минус-цепь: обратный комплемент)
        okc = ok if st == 0 else ok[::-1]
        pos = np.arange(1, L - 1); init = np.isin(s[pos - 1], (1, 3)) & np.isin(s[pos], (0, 2)) & okc[pos]
        pos = pos[init]
        if len(pos) == 0: continue
        pos = pos[rng.random(len(pos)) < RATE]
        if len(pos) == 0: continue
        y = np.zeros(L, np.float32)
        if tr is not None:
            sel = tr["strand"] == st
            q = tr["pos"][sel] if st == 0 else L - 1 - tr["pos"][sel]; y[q] = tr["tot"][sel]
        PG = cs((s == 1) | (s == 2)); PA = cs(s == 0); PT = cs(s == 3); PC = cs(s == 1)
        PCG = cs((s[:-1] == 1) & (s[1:] == 2)); PCG = np.concatenate([PCG, PCG[-1:]])
        ncl = {}
        F = {"part": np.full(len(pos), part[c]), "y": y[pos], "contig_len": np.full(len(pos), L)}
        for k in range(-5, 6): F[f"c{k}"] = s[np.clip(pos + k, 0, L - 1)]
        for w in WIN:
            for name, lo, hi in (("up", pos - MASK - w, pos - MASK), ("dn", pos + MASK + 1, pos + MASK + 1 + w)):
                lo_c, hi_c = np.clip(lo, 0, L), np.clip(hi, 0, L); n = np.maximum(hi_c - lo_c, 1)
                gc = side(PG, lo_c, hi_c) / n; cg = side(PCG, lo_c, np.maximum(hi_c - 1, lo_c)); c_ = side(PC, lo_c, hi_c); g_ = side(PG, lo_c, hi_c) - c_
                F[f"{name}{w}_gc"] = np.where(n < w / 2, np.nan, gc); F[f"{name}{w}_cpg"] = np.where(n < w / 2, np.nan, cg * n / np.maximum(c_ * g_, 1))
                F[f"{name}{w}_a"] = side(PA, lo_c, hi_c) / n; F[f"{name}{w}_t"] = side(PT, lo_c, hi_c) / n
        rows.append(pd.DataFrame(F))
D = pd.concat(rows, ignore_index=True); print("кандидатов", len(D), "положительных", int((D.y > 0).sum()), f"{(time.time() - t0) / 60:.1f} мин", flush=True)
cen = [f"c{k}" for k in range(-5, 6)]; ctx = [c for c in D.columns if c[:2] in ("up", "dn")] + ([] if os.environ.get("EPIC_NO_LEN") == "1" else ["contig_len"])
tr_, ev = D[D.part == 0], D[D.part == 1]; res = []
for name, cols in (("центр", cen), ("окружение", ctx), ("оба", cen + ctx)):
    mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    mdl.fit(tr_[cols], (tr_.y > 0).astype(int)); p = mdl.predict_proba(ev[cols])[:, 1]
    row = dict(признаки=name, AUROC=roc_auc_score(ev.y > 0, p), AUPRC=average_precision_score(ev.y > 0, p),
               AUPRC_tot5=average_precision_score(ev.y >= 5, p), prevalence=float((ev.y > 0).mean()))
    res.append(row); print(row, flush=True)
    if name == "окружение":
        imp = pd.Series(mdl.feature_importances_, cols).sort_values(ascending=False); print(imp.head(12).to_string())
pd.DataFrame(res).to_csv(f"/kaggle/working/ctx_result_m{MASK}.csv", index=False)
print("готово", f"{(time.time() - t0) / 60:.1f} мин")
