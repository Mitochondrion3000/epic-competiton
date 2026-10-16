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
ok = (m1 | m2) & np.isfinite(p); z = (p - p[ok].mean()) / p[ok].std(); z[~ok] = z[ok].min()
pos = (gt > 0) & ok
R = {"n_ok": int(ok.sum()), "n_pos": int(pos.sum())}
# 1. описательная статистика по цепям
st = {}
for nm, s in (("plus", ~minus), ("minus", minus)):
    o = ok & s; q = pos & s
    st[nm] = dict(n_scored=int(o.sum()), n_starts=int(q.sum()), start_rate=float(q.sum() / o.sum()), mean_z_all=float(z[o].mean()), mean_z_starts=float(z[q].mean()),
                  p99_z=float(np.quantile(z[o], 0.99)), mean_label_starts=float(gt[q].mean()), median_label_starts=float(np.median(gt[q])), z_p90_starts=float(np.quantile(z[q], 0.9)))
R["stats"] = st
# 2. метрики: глобальный ранг (как в сабмите), срез по цепи; и ранг внутри цепи
rows = []
for al in (0.6, 0.75):
    for hn, hm in (("half1", m1), ("half2", m2), ("both", m1 | m2)):
        h = hm & ok
        idx = np.flatnonzero(h); rg = np.zeros(len(z), np.float32); rg[idx] = rank_power(z[idx], al)
        for nm, s in (("all", np.ones(len(z), bool)), ("plus", ~minus), ("minus", minus)):
            ii = np.flatnonzero(h & s); rows.append(dict(variant="global_rank", alpha=al, part=hn, strand=nm, **official_score(rg[ii], gt[ii])))
        # ранг отдельно внутри цепи
        rs = np.zeros(len(z), np.float32)
        for s in (~minus, minus):
            ii = np.flatnonzero(h & s); rs[ii] = rank_power(z[ii], al)
        for nm, s in (("all", np.ones(len(z), bool)), ("plus", ~minus), ("minus", minus)):
            ii = np.flatnonzero(h & s); rows.append(dict(variant="rank_inside_strand", alpha=al, part=hn, strand=nm, **official_score(rs[ii], gt[ii])))
# 3. поправки: сдвиг скора минус-цепи так, чтобы совпали квантиль 99% и среднее
def shifted(kind):
    zz = z.copy()
    if kind == "mean": zz[minus] += z[ok & ~minus].mean() - z[ok & minus].mean()
    elif kind == "q99": zz[minus] += np.quantile(z[ok & ~minus], .99) - np.quantile(z[ok & minus], .99)
    return zz
for kind in ("mean", "q99"):
    zz = shifted(kind)
    for al in (0.6, 0.75):
        for hn, hm in (("half1", m1), ("half2", m2), ("both", m1 | m2)):
            idx = np.flatnonzero(hm & ok); r = rank_power(zz[idx], al)
            rows.append(dict(variant=f"shift_{kind}", alpha=al, part=hn, strand="all", **official_score(r, gt[idx])))
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/strand_check_nem.csv", index=False)
json.dump(R, open(f"{OUT}/strand_check_nem.json", "w"), indent=1)
print(json.dumps(R, indent=1)); print(tab[tab.part == "both"].round(4).to_string())
print(tab[tab.part != "both"].pivot_table(index=["variant", "strand"], columns=["alpha", "part"], values=["AUPRC", "Spearman"]).round(4).to_string())
