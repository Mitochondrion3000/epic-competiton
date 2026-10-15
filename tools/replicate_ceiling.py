# Потолок качества по воспроизводимости: насколько реплика r1 «предсказывает» r2 (и наоборот) на тех же метриках EPIC.
# Kaggle CPU-скрипт. Вид Ofas_2.0, подвыборка контигов whitelist (~150 Мб), обе цепи. Выход: /kaggle/working/replicate_ceiling.json
import os, glob, json, tarfile, requests, numpy as np, pandas as pd
SP = "Ofas_2.0"; D = "/tmp/epic"; B = f"{D}/{SP}"; FP = f"{B}/for_participants"
if not os.path.isdir(B):
    os.makedirs(D, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == f"{SP}.tgz"][0]; dst = f"{D}/{SP}.tgz"
    with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(dst, "wb") as fh:
        for ch in r.iter_content(1 << 22): fh.write(ch)
    with tarfile.open(dst) as tf: tf.extractall(D)
    os.remove(dst)
wl = pd.read_csv(f"{FP}/whitelist.train.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], dtype={"chrom": str})
bp = wl.assign(bp=wl.end - wl.start).groupby("chrom").bp.sum(); rng = np.random.default_rng(0)
cs = [c for c in rng.permutation(bp.index[bp >= 100_000]) ]; sel, tot = [], 0
for c in cs:
    if tot >= 150e6: break
    sel.append(c); tot += bp[c]
print("контигов", len(sel), "Мб", tot / 1e6)
files = sorted(glob.glob(f"{FP}/csRNA-*r[0-9]-train.bed.gz")); assert len(files) >= 2, files
wls = wl[wl.chrom.isin(set(sel))]; clen = wls.groupby("chrom").end.max()
def dense(path):
    d = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 4, 5], names=["chrom", "start", "end", "count", "strand"], dtype={"chrom": str, "strand": str})
    d = d[d.chrom.isin(set(sel))]; out = {}
    for (c, st), g in d.groupby(["chrom", "strand"]):
        a = np.zeros(int(clen[c]) + 1, np.float32)
        diff = np.zeros(len(a) + 1, np.float32); np.add.at(diff, g.start.values.clip(0, len(a)), g["count"].values); np.add.at(diff, g.end.values.clip(0, len(a)), -g["count"].values)
        out[(c, st)] = np.cumsum(diff)[:len(a)]
    return out
R1, R2 = dense(files[0]), dense(files[1])
def vec(R):
    parts = []
    for c in sel:
        m = np.zeros(int(clen[c]) + 1, bool)
        for s, e in zip(wls[wls.chrom == c].start.values, wls[wls.chrom == c].end.values): m[s:e] = True
        for st in ("+", "-"): parts.append(R.get((c, st), np.zeros(len(m), np.float32))[m])
    return np.concatenate(parts)
a, b = vec(R1), vec(R2); print("позиций", len(a), "стартов r1", int((a > 0).sum()), "r2", int((b > 0).sum()), "общих", int(((a > 0) & (b > 0)).sum()))
SCALE = 100_000
def quantise(p): return np.rint(np.clip(p, 0, 1) * SCALE).astype(np.int32)
def prauc(lv, y):
    pos = np.bincount(lv[y], minlength=SCALE + 1)[::-1].astype(np.int64); tot = np.bincount(lv, minlength=SCALE + 1)[::-1].astype(np.int64)
    keep = tot > 0; tp, n = np.cumsum(pos)[keep], np.cumsum(tot)[keep]; P = tp[-1]
    return float((np.diff(np.r_[0, tp]) / P * (tp / n)).sum())
def dr(x): return np.unique(x, return_inverse=True)[1].astype(np.float64)
def spear(p, g):
    x, y = dr(p) - dr(p).mean(), dr(g) - dr(g).mean(); return float(x @ y / np.sqrt((x * x).sum() * (y * y).sum()))
res = {}
for nm, p, g in (("r1->r2", a, b), ("r2->r1", b, a)):
    y = g > 0; lv = quantise(np.minimum(p, SCALE) / SCALE); res[nm] = dict(AUPRC=prauc(lv, y), Spearman=spear(lv[y], g[y]), n_starts=int(y.sum()), n_pos=len(p))
res["mean"] = {k: (res["r1->r2"][k] + res["r2->r1"][k]) / 2 for k in ("AUPRC", "Spearman")}
print(json.dumps(res, indent=1)); json.dump(res, open("/kaggle/working/replicate_ceiling.json", "w"))
