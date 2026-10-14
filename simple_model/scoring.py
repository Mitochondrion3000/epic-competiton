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


def write_submission(values01, path, chunk=8_000_000):
    """Каждая строка ровно 'd.ddddd\\n', как '%.5f', но быстро."""
    ints = np.rint(np.clip(values01, 0, 1) * SCALE).astype(np.int64)
    with gzip.open(path, "wb", compresslevel=4) as fh:
        for a in range(0, len(ints), chunk):
            v = ints[a:a + chunk]; buf = np.empty((len(v), 8), np.uint8)
            buf[:, 0] = ord("0") + v // SCALE; buf[:, 1] = ord("."); buf[:, 7] = ord("\n")
            frac = v % SCALE
            for k in range(6, 1, -1): buf[:, k] = ord("0") + frac % 10; frac //= 10
            fh.write(buf.tobytes())
