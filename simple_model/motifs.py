"""Мотивы факторов транскрипции из баз JASPAR и HOCOMOCO: чтение, скрининг около стартов, признаки.

Мотив задан матрицей частот (PFM): для каждой из w позиций вероятность букв A, C, G, T. Переводим её в веса
log2(частота / фон) (PWM). Счёт участка длины w = сумма весов его букв; чем выше, тем больше участок похож на сайт
связывания фактора. Фактор садится на двойную спираль, поэтому смотрим обе ориентации и берём максимум.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

URLS = {
    "hocomoco12": "https://hocomoco12.autosome.org/final_bundle/hocomoco12/H12CORE/formatted_motifs/H12CORE_meme_format.meme",
    **{f"jaspar_{t}": f"https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_{t}_non-redundant_pfms_meme.txt"
       for t in ("vertebrates", "insects", "nematodes", "urochordates")},
}


def read_meme(path, source):
    """MEME-файл -> список (имя, фактор, PFM w x 4). Имя фактора: для HOCOMOCO до первой точки, для JASPAR второе слово."""
    out, name, tf, rows = [], None, None, []
    def flush():
        if rows: out.append((name, tf, np.array(rows)))
    for line in open(path):
        if line.startswith("MOTIF"):
            flush(); rows = []
            parts = line.split()
            if source.startswith("jaspar"):
                name, tf = f"{parts[2]} ({parts[1]})", parts[2]
            else:
                name, tf = parts[1], parts[1].split(".")[0]
        elif name and len(line.split()) == 4 and line.strip()[0] in "0123456789.":
            rows.append([float(x) for x in line.split()])
    flush()
    return out


def pwm_from_pfm(pfm, bg, pseudo=0.01):
    p = (pfm + pseudo) / (1 + 4 * pseudo)
    return np.log2(p / np.asarray(bg)).astype(np.float32)


def onehot(W):
    """Коды (n x L, 0..4) -> one-hot (n x L x 4); N даёт нули."""
    X = np.zeros(W.shape + (4,), np.float32)
    for b in range(4): X[..., b] = (W == b)
    return X


def window_scores(X, pwms, lo, hi, chunk=1000):
    """Для окон X (n x L x 4) и набора PWM одной длины (G x w x 4): счёт (max по двум ориентациям) для каждого
    начала мотива в [lo, hi) (индексы колонок окна). Возвращает массив n x (hi-lo) x G."""
    G, w, _ = pwms.shape
    Wf = pwms.reshape(G, -1).T; Wr = pwms[:, ::-1, ::-1].reshape(G, -1).T      # прямая и обратно-комплементарная
    out = []
    for c in range(0, len(X), chunk):
        V = sliding_window_view(X[c:c + chunk], (w, 4), axis=(1, 2))[:, lo:hi, 0].reshape(-1, w * 4)
        V = np.ascontiguousarray(V)
        out.append(np.maximum(V @ Wf, V @ Wr).reshape(-1, hi - lo, G))
    return np.concatenate(out)


def score_track(o, pwm):
    """Счёт мотива (max по двум ориентациям) для каждого индекса последовательности o как начала сайта."""
    L, w = len(o), len(pwm)
    rc = pwm[::-1, ::-1]
    f = np.zeros(L, np.float32); r = np.zeros(L, np.float32)
    pad = np.concatenate([pwm, np.zeros((w, 1), np.float32)], 1)     # колонка 4 = N, вес 0
    padr = np.concatenate([rc, np.zeros((w, 1), np.float32)], 1)
    for k in range(w):
        f[:L - k] += pad[k][o[k:]]; r[:L - k] += padr[k][o[k:]]
    s = np.maximum(f, r)
    s[L - w + 1:] = -1e9
    return s


def max_scores(X, pwms, lo, hi, chunk=1000):
    """Как window_scores, но сразу максимум по окну: n x G (лучший сайт каждого мотива в окне)."""
    G, w, _ = pwms.shape
    Wf = pwms.reshape(G, -1).T; Wr = pwms[:, ::-1, ::-1].reshape(G, -1).T
    out = []
    for c in range(0, len(X), chunk):
        V = sliding_window_view(X[c:c + chunk], (w, 4), axis=(1, 2))[:, lo:hi, 0]
        n = V.shape[0]; V = np.ascontiguousarray(V.reshape(-1, w * 4))
        out.append(np.maximum(V @ Wf, V @ Wr).reshape(n, hi - lo, G).max(1))
    return np.concatenate(out)


def site_thresholds(X_bg, pwms, q=0.9999, chunk=500):
    """Порог счёта для каждого мотива: квантиль q счёта во всех позициях фоновых окон.
    q = 0.9999 значит, что в случайной ДНК сайт с таким счётом встречается примерно раз на 10 000 позиций."""
    G, w, _ = pwms.shape
    Wf = pwms.reshape(G, -1).T; Wr = pwms[:, ::-1, ::-1].reshape(G, -1).T
    parts = []
    for c in range(0, len(X_bg), chunk):
        V = sliding_window_view(X_bg[c:c + chunk], (w, 4), axis=(1, 2))[:, :, 0]
        V = np.ascontiguousarray(V.reshape(-1, w * 4))
        parts.append(np.maximum(V @ Wf, V @ Wr))
    return np.quantile(np.concatenate(parts), q, axis=0)


def kmer_best_match(kmer, motifs, bg, top=3, min_ratio=0.9):
    """С какими мотивами базы совпадает k-мер. Для каждого мотива ищем место и ориентацию, где k-мер набирает не меньше
    min_ratio от максимально возможного счёта, и среди таких мест берём самое информативное (наибольший счёт в битах):
    короткий k-мер «помещается» во многие длинные мотивы, но осмысленно только совпадение с их ядром."""
    idx = ["ACGT".index(c) for c in kmer]; k = len(kmer)
    res = []
    for name, tf, pfm in motifs:
        pwm = pwm_from_pfm(pfm, bg)
        best = -np.inf
        for P in (pwm, pwm[::-1, ::-1]):
            for s in range(0, len(P) - k + 1):
                sub = P[s:s + k]
                got, mx = sub[np.arange(k), idx].sum(), sub.max(1).sum()
                if mx > 0 and got >= min_ratio * mx:
                    best = max(best, got)
        if best > -np.inf:
            res.append((float(best), tf, name))
    res.sort(reverse=True)
    return res[:top]
