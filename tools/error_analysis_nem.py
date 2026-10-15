# Kaggle-скрипт (CPU): разбор ошибок на ТЕСТЕ Nematostella jaNemVect1.1 (метки теста открыты) для готовых предсказаний.
# Вход: EPIC_PRED (по умолчанию ищет *simple_model_final.predictions.txt.gz в /kaggle/input). Выход: /kaggle/working/error_analysis_nem.csv, error_positions_nem.csv.gz
import os, glob, gzip, hashlib, tarfile, numpy as np, pandas as pd, requests
from scipy.stats import mannwhitneyu
SP = "jaNemVect1.1"; D = "/tmp/epic"; B = f"{D}/{SP}"; OUT = "/kaggle/working"; CAP = 20000
if not os.path.isdir(B):
    os.makedirs(D, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == f"{SP}.tgz"][0]; dst = f"{D}/{SP}.tgz"
    with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(dst, "wb") as fh:
        for ch in r.iter_content(1 << 22): fh.write(ch)
    with tarfile.open(dst) as tf: tf.extractall(D)
    os.remove(dst)
pf = os.environ.get("EPIC_PRED") or glob.glob(f"/kaggle/input/**/{SP}.simple_model_final.predictions.txt.gz", recursive=True)[0]
rd = lambda p, dt: pd.read_csv(p, header=None, dtype=dt)[0].values
p = rd(pf, np.float32); gt = rd(f"{B}/ground_truth/csRNA-test.txt.gz", np.float32)
m1 = rd(f"{B}/scoring/mask1-test.txt.gz", np.int8).astype(bool); m2 = rd(f"{B}/scoring/mask2-test.txt.gz", np.int8).astype(bool)
assert len(p) == len(gt), (len(p), len(gt))
tpl = pd.read_csv(f"{B}/for_participants/template.test.bed.gz", sep="\t", header=None, usecols=range(6), names=["chrom","start","end","name","score","strand"], dtype={"chrom": str, "strand": str})
ln = (tpl.end - tpl.start).values; off = np.r_[0, np.cumsum(ln)[:-1]]; assert ln.sum() == len(p)
LUT = np.full(256, 8, np.uint8)
for i, b in enumerate("ACGT"): LUT[ord(b)] = i; LUT[ord(b.lower())] = i + 4
genome, name, ch = {}, None, []
keep = set(tpl.chrom)
def flush():
    if name in keep: genome[name] = LUT[np.frombuffer(b"".join(ch), np.uint8)]
with gzip.open(f"{B}/genome/genome.fa.gz", "rb") as fh:
    for l in fh:
        if l.startswith(b">"): flush(); name = l[1:].split()[0].decode(); ch = []
        elif name in keep: ch.append(l.rstrip())
flush()
ok = m1 | m2                                  # позиции, которые участвуют в скоринге
pos = (gt > 0) & ok; N = int(pos.sum()); thr = np.partition(p[ok], -N)[-N]; top = (p >= thr) & ok
grp = np.full(len(gt), "", object); grp[pos & top] = "TP"; grp[pos & ~top] = "FN"; grp[~pos & top] = "FP"
rng = np.random.default_rng(0)
for g in ("TP", "FN", "FP"):
    ix = np.flatnonzero(grp == g)
    if len(ix) > CAP: grp[rng.choice(ix, len(ix) - CAP, replace=False)] = ""
grp[rng.choice(np.flatnonzero(ok & ~top & ~pos), CAP, replace=False)] = "BG"
print("стартов", N, {g: int((grp == g).sum()) for g in ("TP","FN","FP","BG")}, "доля FN от стартов", ((pos & ~top).sum() / N).round(3))
idx = np.flatnonzero(grp != ""); rid = np.searchsorted(off, idx, side="right") - 1
chrom = tpl.chrom.values[rid]; gpos = tpl.start.values[rid] + (idx - off[rid]); strand = (tpl.strand.values[rid] == "-").astype(int)
COMP = np.array([3, 2, 1, 0, 4, 5, 6, 7, 8])
def window(c, g, st, h=100):
    s = genome[c]; a = s[max(g - h, 0):g + h + 1]; a = np.pad(a, (max(h - g, 0), max(g + h + 1 - len(s), 0)), constant_values=8)
    return COMP[a[::-1]] if st else a
recs = []
for c, g, st, i in zip(chrom, gpos, strand, idx):
    w = window(c, int(g), int(st)); u = w % 4; low = (w >= 4) & (w < 8); o = w != 8
    cc, gg = np.sum((u == 1) & o), np.sum((u == 2) & o); cg = np.sum((u[:-1] == 1) & (u[1:] == 2) & o[:-1] & o[1:])
    k3 = u[:-2] * 16 + u[1:-1] * 4 + u[2:]; cnt = np.bincount(k3[o[:-2]], minlength=64) + 0.0; q = cnt / max(cnt.sum(), 1)
    recs.append(dict(i=i, grp=grp[i], gt=gt[i], pred=p[i], chrom=c, strand=st, gc=(cc + gg) / max(o.sum(), 1), cpg_oe=cg / max(cc * gg / max(o.sum(), 1), 1e-9),
        lower=low.mean(), ent3=-(q[q > 0] * np.log2(q[q > 0])).sum(), dinuc="ACGTN"[w[99] % 4 if w[99] < 8 else 4] + "ACGTN"[w[100] % 4 if w[100] < 8 else 4]))
df = pd.DataFrame(recs); pidx = np.flatnonzero(pos)
df["n_starts_pm50"] = [int(np.searchsorted(pidx, i + 51) - np.searchsorted(pidx, i - 50)) - int(gt[i] > 0) for i in df.i]
df.to_csv(f"{OUT}/error_positions_nem.csv.gz", index=False)
num = ["gc", "cpg_oe", "lower", "ent3", "n_starts_pm50", "gt", "pred"]
s = df.groupby("grp")[num].median().T
for g, ref in (("FN", "TP"), ("FP", "BG")):
    for f in num[:5]: s.loc[f, f"p({g} vs {ref})"] = mannwhitneyu(df[df.grp == g][f].dropna(), df[df.grp == ref][f].dropna()).pvalue
s.to_csv(f"{OUT}/error_analysis_nem.csv"); print(s.round(4).to_string())
print(pd.crosstab(df.dinuc, df.grp, normalize="columns").round(3).sort_values("TP", ascending=False).head(10).to_string())
df["gt_bin"] = pd.cut(df.gt, [-1, 0, 1, 2, 4, 9, 49, 1e9]); t = df[df.grp.isin(["FN", "TP"])]
print("доля FN по силе:\n", t.groupby("gt_bin", observed=True).grp.apply(lambda x: (x == "FN").mean()).round(3).to_string())
print("доля FN по контигам:\n", t.groupby("chrom").grp.apply(lambda x: (x == "FN").mean()).round(3).head(15).to_string())
