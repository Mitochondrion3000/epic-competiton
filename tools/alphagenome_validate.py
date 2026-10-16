"""D-1: валидация AlphaGenome (API) на Nematostella: есть ли сигнал инициации в предсказанных PROCAP/CAGE для чужого вида.

Идея: окна 1 Мнт (SEQUENCE_LENGTH_1MB) по тестовым контигам -> predict_sequence(последовательность, HOMO_SAPIENS, PROCAP/CAGE) -> скор позиции = среднее
(или максимум) по трекам нужной цепи -> официальный скор против настоящей csRNA-seq. Критерий (D-1 п. 3): AUPRC >= 0.01 (случайный 0.0012) = сигнал есть.
Ключ API только из окружения ALPHAGENOME_API_KEY (или секрет Kaggle `ALPHAGENOME_API_KEY`); в файлы и логи не пишется.
Бюджет: --budget-mb (20-50 Мбнт = 20-50 запросов). Результаты: ag_nem_scores.npz (кэш предсказаний) и ag_nem_metrics.csv.
Здесь только логика; драйвер для Kaggle собирается в kaggle/alphagenome_nem/run.py (make.py).
"""
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
