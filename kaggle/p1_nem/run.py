import numpy as np
"""Официальные метрики EPIC и запись сабмита (та же логика, что в epic_v2_model.ipynb, раздел 6.1 и 8.3).

* Сабмит: по одному числу в [0, 1] с 5 знаками после запятой на каждую позицию шаблона, поэтому скорер видит не больше
  100 001 различных уровней.
* AUPRC = average precision по всем позициям (равные значения идут одной группой).
* Spearman = корреляция Пирсона между *плотными* рангами (dense rank: равные значения получают один ранг, ранги идут
  подряд без пропусков) предсказания и правды, только в позициях, где правда > 0.
"""
import gzip
import numpy as np

SCALE = 100_000


def quantise(pred01):
    return np.rint(np.clip(pred01, 0.0, 1.0) * SCALE).astype(np.int32)


def prauc_from_levels(levels, labels):
    pos = np.bincount(levels[labels], minlength=SCALE + 1)[::-1].astype(np.int64)
    tot = np.bincount(levels, minlength=SCALE + 1)[::-1].astype(np.int64)
    keep = tot > 0; tp, n = np.cumsum(pos)[keep], np.cumsum(tot)[keep]
    P = tp[-1] if len(tp) else 0
    if P == 0: return 0.0
    return float((np.diff(np.r_[0, tp]) / P * (tp / n)).sum())


def dense_rank(x):
    return np.unique(x, return_inverse=True)[1].astype(np.float64)


def spearman_dense(pred_pos, gt_pos):
    if len(pred_pos) < 2: return 0.0
    a, b = dense_rank(pred_pos), dense_rank(gt_pos)
    a -= a.mean(); b -= b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if d == 0 else float(a @ b / d)


def official_score(pred01, gt, mask=None):
    """Метрики так, как их считает скорер по файлу сабмита (pred01 — значения в [0, 1])."""
    if mask is not None: pred01, gt = pred01[mask], gt[mask]
    lv, y = quantise(pred01), gt > 0
    return {"AUPRC": prauc_from_levels(lv, y), "Spearman": spearman_dense(lv[y], gt[y])}


def rank_power(score, alpha, nmask=None):
    """Монотонно переводим любой счёт в [0, 1]: позиция с местом r в рейтинге (1 = лучшая) получает r^(-alpha).
    Порядок позиций не меняется, но при округлении до 5 знаков все позиции ниже некоторого места сливаются в одно
    значение (и в один плотный ранг). alpha задаёт, сколько верхних мест сохраняют различимость. N-позиции -> 0."""
    x = score.astype(np.float32, copy=True)
    if nmask is not None: x[nmask] = -np.inf
    order = np.argsort(-x, kind="stable")
    ranks = np.empty(len(x), np.int32); ranks[order] = np.arange(1, len(x) + 1, dtype=np.int32); del order
    s = (1.0 - (ranks - 1) / len(x)) if alpha == 0 else np.power(ranks.astype(np.float32), -alpha)
    s = np.rint(np.clip(s, 0, 1) * SCALE).astype(np.float32) / SCALE
    if nmask is not None: s[nmask] = 0.0
    return s



import itertools, json, numpy as np
from scipy.ndimage import uniform_filter1d
ALPHAS = [0.6, 0.75, 0.9]

def local_norm(z, beta, R, seg=None):
    """P-1: score - beta * (локальное среднее score в ±R). seg=(off, len) — среднее считается внутри каждой строки (контиг/цепь), не через границы."""
    if seg is None: return (z - beta * uniform_filter1d(z, 2 * R + 1, mode="nearest")).astype(np.float32)
    out = np.empty_like(z, dtype=np.float32)
    for o, l in zip(*seg): out[o:o + l] = z[o:o + l] - beta * uniform_filter1d(z[o:o + l], 2 * R + 1, mode="nearest")
    return out

def run_p1(z, gt, ok, fold, out, seg=None, Rs=(2000, 5000, 10000, 20000), betas=(0.25, 0.5, 0.75, 1.0), alphas=ALPHAS, max_sp_loss=0.01):
    """P-1: делительная/вычитающая локальная нормировка. Параметры (R, beta, alpha) выбираются на одной половине (макс. AUPRC при потере
    Spearman <= max_sp_loss против лучшего baseline этой половины), оценка на другой; обе стороны. Критерий приёмки: AUPRC +3% при потере Spearman < 0.01."""
    import pandas as pd
    z = z.astype(np.float32); idx = {f: np.flatnonzero((fold == f) & ok) for f in (0, 1)}; rows = []
    def measure(R, beta, s):
        for a in alphas:
            for f in (0, 1):
                m = official_score(rank_power(s[idx[f]], a), gt[idx[f]]); rows.append(dict(R=R, beta=beta, alpha=a, fold=f, **m))
    measure(0, 0.0, z)
    for R, b in itertools.product(Rs, betas): measure(R, b, local_norm(z, b, R, seg)); print("R", R, "beta", b, flush=True)
    df = pd.DataFrame(rows); df.to_csv(out + ".grid.csv", index=False); res = []
    for ev in (0, 1):
        tr = 1 - ev; g = df[df.fold == tr]; bt = g[g.R == 0].sort_values("AUPRC", ascending=False).iloc[0]
        cand = g[(g.R > 0) & (g.Spearman >= bt.Spearman - max_sp_loss)]
        pick = (cand if len(cand) else g[g.R > 0]).sort_values("AUPRC", ascending=False).iloc[0]
        be = df[(df.fold == ev) & (df.R == 0) & (df.alpha == bt.alpha)].iloc[0]
        pe = df[(df.fold == ev) & (df.R == pick.R) & (df.beta == pick.beta) & (df.alpha == pick.alpha)].iloc[0]
        res.append(dict(eval_fold=ev, R=pick.R, beta=pick.beta, alpha=pick.alpha, base_alpha=bt.alpha, AUPRC_base=be.AUPRC, AUPRC=pe.AUPRC,
                        Spearman_base=be.Spearman, Spearman=pe.Spearman, dAUPRC_pct=(pe.AUPRC / be.AUPRC - 1) * 100, dSpearman=pe.Spearman - be.Spearman))
    R_ = pd.DataFrame(res); R_.to_csv(out + ".summary.csv", index=False); print(R_.round(4).to_string())
    print("СРЕДНЕЕ dAUPRC_pct = %.2f, dSpearman = %.4f" % (R_.dAUPRC_pct.mean(), R_.dSpearman.mean()))
    return R_

import os, glob, json, tarfile, requests, pandas as pd
SP = "jaNemVect1.1"; D = "/tmp/epic"; B = f"{D}/{SP}"; OUT = "/kaggle/working"
if not os.path.isdir(B):
    os.makedirs(D, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == f"{SP}.tgz"][0]; dst = f"{D}/{SP}.tgz"
    with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(dst, "wb") as fh:
        for ch in r.iter_content(1 << 22): fh.write(ch)
    with tarfile.open(dst) as tf: tf.extractall(D)
    os.remove(dst)
rd = lambda p, dt: pd.read_csv(p, header=None, dtype=dt)[0].values
gt = rd(f"{B}/ground_truth/csRNA-test.txt.gz", np.float32)
m1 = rd(f"{B}/scoring/mask1-test.txt.gz", np.int8).astype(bool); m2 = rd(f"{B}/scoring/mask2-test.txt.gz", np.int8).astype(bool)
tpl = pd.read_csv(f"{B}/for_participants/template.test.bed.gz", sep="\t", header=None, usecols=range(6), names=["chrom","start","end","name","score","strand"], dtype={"chrom": str, "strand": str})
ln = (tpl.end - tpl.start).values; assert ln.sum() == len(gt)
minus = np.repeat((tpl.strand.values == "-"), ln)
cand = glob.glob("/kaggle/input/**/test_lograte.npy", recursive=True); print(cand)
p = np.load([c for c in cand if SP in c][0] if any(SP in c for c in cand) else cand[0]).astype(np.float32).ravel()
assert len(p) == len(gt), (len(p), len(gt))
ok = (m1 | m2) & np.isfinite(p); z = (p - p[ok].mean()) / p[ok].std(); z[~ok] = z[ok].min(); z = z.astype(np.float32)
fold = np.where(m1, 0, np.where(m2, 1, -1)).astype(np.int8)
off = np.r_[0, np.cumsum(ln)[:-1]]
S = run_p1(z, gt, ok, fold, "/kaggle/working/p1_nem", seg=(off, ln))
