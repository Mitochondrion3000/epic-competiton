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




import numpy as np
import os
import numpy as np

W = 1_048_576                                  # SEQUENCE_LENGTH_1MB
TYPES = ("PROCAP", "CAGE")


def make_client():
    from alphagenome.models import dna_client
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient; key = UserSecretsClient().get_secret("ALPHAGENOME_API_KEY")
        except Exception: pass
    assert key, "нет ключа: задайте ALPHAGENOME_API_KEY (https://alphagenome.google/api)"
    return dna_client.create(key)


def ag_predict(client, seq, types=TYPES):
    """seq: строка ровно из W букв ACGTN. Возвращает {тип: (values (W, n_tracks) float32, strands (n_tracks,) из '+', '-', '.')}"""
    from alphagenome.models import dna_client
    out = client.predict_sequence(seq, organism=dna_client.Organism.HOMO_SAPIENS,
                                  requested_outputs=[getattr(dna_client.OutputType, t) for t in types], ontology_terms=None)
    res = {}
    for t in types:
        td = getattr(out, t.lower())
        res[t] = (np.asarray(td.values, np.float32), np.asarray(td.metadata["strand"].values if "strand" in td.metadata else ["."] * td.values.shape[1]))
    return res


def strand_scores(pred, how="mean"):
    """{тип: (values, strands)} -> {(тип, how, цепь): (W,)}; цепь 0 = '+', 1 = '-'"""
    out = {}
    for t, (v, st) in pred.items():
        for k, sym in ((0, "+"), (1, "-")):
            sel = st == sym
            if not sel.any(): sel = st == "."                                 # трек без цепи: одинаковый для обеих
            if not sel.any(): continue
            out[(t, how, k)] = (v[:, sel].mean(1) if how == "mean" else v[:, sel].max(1)).astype(np.float32)
    return out


def predict_windows(client, genome, windows, cache=None, types=TYPES, predict=ag_predict):
    """windows: [(chrom, start)], start кратен W. genome[chrom] — коды 0..3 ACGT (+4 строчные), 8 = N. Возвращает {(chrom, start): {(тип, how, цепь): (W,)}}"""
    LUT = np.array(list("ACGTACGTN"))
    res = {} if cache is None else cache
    for c, s in windows:
        if (c, s) in res: continue
        g = genome[c][s:s + W]; seq = "".join(LUT[g]).ljust(W, "N")
        p = predict(client, seq, types); d = {}
        for how in ("mean", "max"): d.update(strand_scores(p, how))
        res[(c, s)] = d
    return res


def build_scores(tpl, windows_pred, key):
    """Разворачиваем предсказания окон в массив по позициям шаблона (как gt): позиции вне посчитанных окон -> nan."""
    ln = (tpl["end"] - tpl["start"]).values; off = np.r_[0, np.cumsum(ln)[:-1]]
    sc = np.full(int(ln.sum()), np.nan, np.float32)
    for c, s0, e0, st, o in zip(tpl["chrom"].values, tpl["start"].values, tpl["end"].values, (tpl["strand"].values == "-").astype(int), off):
        for w0 in range(s0 // W * W, e0, W):
            d = windows_pred.get((c, w0))
            if d is None or (key[0], key[1], st) not in d: continue
            a, b = max(s0, w0), min(e0, w0 + W)
            sc[o + a - s0:o + b - s0] = d[(key[0], key[1], st)][a - w0:b - w0]
    return sc


def select_windows(tpl, mask_rows, budget):
    """Первые `budget` окон 1 Мнт, пересекающих строки шаблона с тестовой разметкой (mask_rows — bool на строку)."""
    seen = []
    for c, s0, e0, m in zip(tpl["chrom"].values, tpl["start"].values, tpl["end"].values, mask_rows):
        if not m: continue
        for w0 in range(s0 // W * W, e0, W):
            if (c, w0) not in seen: seen.append((c, w0))
            if len(seen) >= budget: return seen
    return seen


def evaluate(scores, gt, ok, official_score, rank_power, alphas=(0.6, 0.75)):
    """scores с nan вне покрытия; метрики на покрытых позициях ok. Случайный AUPRC = доля положительных."""
    cov = ok & np.isfinite(scores); n = int(cov.sum())
    if n == 0: return []
    z = scores[cov]; g = gt[cov]; base = float((g > 0).mean())
    return [dict(alpha=a, n=n, random_auprc=base, **official_score(rank_power(z, a), g)) for a in alphas]

import glob, gzip, tarfile, requests, pandas as pd
BUDGET = int(os.environ.get("AG_BUDGET", "40")); SP = "jaNemVect1.1"; D = "/tmp/epic"; B = f"{D}/{SP}"; OUT = "/kaggle/working"
os.system("pip install -q alphagenome")
client = make_client(); print("ключ найден, клиент создан", flush=True)   # раньше загрузки данных: без секрета падаем сразу
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
ln = (tpl.end - tpl.start).values; off = np.r_[0, np.cumsum(ln)[:-1]]; assert ln.sum() == len(gt)
ok = m1 | m2
rows_ok = np.array([ok[o:o + l].any() for o, l in zip(off, ln)])
LUT = np.full(256, 8, np.uint8)
for i, b in enumerate("ACGT"): LUT[ord(b)] = i; LUT[ord(b.lower())] = i + 4
genome, name, ch = {}, None, []; keep = set(tpl.chrom)
def flush():
    if name in keep: genome[name] = LUT[np.frombuffer(b"".join(ch), np.uint8)]
with gzip.open(f"{B}/genome/genome.fa.gz", "rb") as fh:
    for l in fh:
        if l.startswith(b">"): flush(); name = l[1:].split()[0].decode(); ch = []
        elif name in keep: ch.append(l.rstrip())
flush()
wins = select_windows(tpl, rows_ok, BUDGET); print("окон:", len(wins), "~Мнт:", len(wins) * W / 1e6, flush=True)
cache = {}
for i in range(0, len(wins), 5):                             # по 5 окон, чтобы не терять прогресс при сбое
    predict_windows(client, genome, wins[i:i + 5], cache); print("готово окон", len(cache), flush=True)
rows = []
for t in TYPES:
    for how in ("mean", "max"):
        s = build_scores(tpl, cache, (t, how))
        for r in evaluate(s, gt, ok, official_score, rank_power): rows.append(dict(type=t, agg=how, **r))
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/ag_nem_metrics.csv", index=False); print(tab.round(5).to_string())

# --- шаг 4 D-1: признак AlphaGenome поверх скора v2 (CPU-реранкер: z_v2 + w * z_ag), w и alpha на одной половине теста, оценка на другой
cand = glob.glob("/kaggle/input/**/test_lograte.npy", recursive=True); print(cand)
p2 = np.load([c for c in cand if SP in c][0] if any(SP in c for c in cand) else cand[0]).astype(np.float32).ravel(); assert len(p2) == len(gt)
S = {(t, how): build_scores(tpl, cache, (t, how)) for t in TYPES for how in ("mean",)}
np.savez_compressed(f"{OUT}/ag_nem_scores.npz", **{f"{t}_{h}": np.nan_to_num(v, nan=-1).astype(np.float16) for (t, h), v in S.items()})
cov = ok & np.isfinite(p2) & np.isfinite(S[("CAGE", "mean")]) & np.isfinite(S[("PROCAP", "mean")])
zs = lambda v: (v - v[cov].mean()) / v[cov].std()
z2 = zs(p2); res = []
for t in TYPES:
    za = zs(np.log1p(np.maximum(S[(t, "mean")], 0)))
    for W_ in (0.0, 0.1, 0.2, 0.35, 0.5, 1.0):
        sc = z2 + W_ * za
        for a_ in (0.6, 0.75):
            for hn, hm in (("half1", m1), ("half2", m2)):
                ii = np.flatnonzero(cov & hm)
                if len(ii): res.append(dict(type=t, w=W_, alpha=a_, half=hn, **official_score(rank_power(sc[ii], a_), gt[ii])))
R2 = pd.DataFrame(res); R2.to_csv(f"{OUT}/ag_nem_combine.csv", index=False); print(R2.round(4).to_string())
for t in TYPES:
    out = []
    for ev, tr in (("half1", "half2"), ("half2", "half1")):
        g = R2[(R2.type == t) & (R2.half == tr)]; best = g.sort_values("AUPRC", ascending=False).iloc[0]
        e = R2[(R2.type == t) & (R2.half == ev)]
        base = e[(e.w == 0.0) & (e.alpha == best.alpha)].iloc[0]; pick = e[(e.w == best.w) & (e.alpha == best.alpha)].iloc[0]
        out.append(dict(type=t, eval=ev, w=best.w, alpha=best.alpha, AUPRC_v2=base.AUPRC, AUPRC_comb=pick.AUPRC, Sp_v2=base.Spearman, Sp_comb=pick.Spearman, dAUPRC_pct=(pick.AUPRC / base.AUPRC - 1) * 100))
    print(pd.DataFrame(out).round(4).to_string())
print("СИГНАЛ ЕСТЬ (AUPRC >= 0.01)" if (tab.AUPRC >= 0.01).any() else "СИГНАЛА НЕТ: AUPRC < 0.01 во всех вариантах")
