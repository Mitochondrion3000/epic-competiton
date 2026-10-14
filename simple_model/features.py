"""Понятные признаки для каждой позиции-цепи, посчитанные только по последовательности ДНК.

Используется в шагах 2-4 серии simple_model. Главная идея реализации: каждый признак считается как «дорожка» вдоль всего
контига (одно число на каждую позицию), поэтому одну и ту же функцию можно применить и к 2.5 млн строк обучающей таблицы,
и ко всем 72 млн позиций теста.

Ориентация. Для цепи «−» работаем с обратно-комплементарной последовательностью контига. В ней позиция генома p имеет
индекс L-1-p, а смещение +k по-прежнему означает «k нт ниже по течению (по ходу транскрипции)». Поэтому все окна ниже
записаны в координатах транскрипции: отрицательные смещения это upstream (промотор), положительные это тело РНК.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d

COMP = np.array([3, 2, 1, 0, 4], np.uint8)          # коды 0..4 = A,C,G,T,N; комплемент A<->T, C<->G
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "W": "AT", "S": "CG", "K": "GT", "M": "AC",
         "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT"}

# ---------------------------------------------------------------------------------------------------------------------
# Каталог признаков. motif: (последовательность IUPAC, допустимых несовпадений, окно начала мотива [a, b] относительно старта)
# Окна выбраны по профилям обогащения в step2 (раздел 2): там видно, где именно мотив встречается чаще, чем в фоне.
# ---------------------------------------------------------------------------------------------------------------------
MOTIFS = {
    "tata":        ("TATAWAWR", 0, -36, -25, "TATA-бокс: место посадки TBP, классически в −30"),
    "ccaat":       ("CCAAT", 0, -150, -40, "CCAAT-бокс (фактор NF-Y), типичный элемент промотора выше старта"),
    "ebox":        ("CACGTG", 0, -100, 20, "E-бокс: сайт факторов bHLH (Myc/Max, USF)"),
    "cre":         ("TGACG", 0, -150, 20, "половина CRE (TGACGTCA): сайт факторов CREB/ATF"),
    "trna_abox":   ("TRGCNNAGYNGG", 1, -50, 30, "A-бокс гена тРНК (промотор РНК-полимеразы III внутри гена)"),
    "trna_bbox":   ("GGTTCGANTCC", 1, -10, 70, "B-бокс гена тРНК"),
    "splice_down": ("GTRAGT", 0, 20, 300, "донорный сайт сплайсинга ниже старта: конец первого экзона"),
    "polyAT_core": (None, 0, -60, 40, "участки AAAAAA/TTTTTT у самого старта (у стартов обеднены)"),
    "polyAT_up":   (None, 0, -300, -120, "участки AAAAAA/TTTTTT далеко выше старта (у стартов обогащены)"),
}
# Состав в окне [a, b]: (что считаем, a, b, описание)
COMPOSITION = {
    "gc_core":   ("gc", -50, 50, "доля G+C вокруг старта"),
    "gc_up":     ("gc", -300, -51, "доля G+C выше старта (промотор)"),
    "gc_down":   ("gc", 51, 300, "доля G+C ниже старта (тело РНК)"),
    "cpg_core":  ("cpg", -60, 60, "CpG наблюдаемое/ожидаемое вокруг старта (CpG-островок)"),
    "cpg_down":  ("cpg", 120, 240, "CpG наблюдаемое/ожидаемое ниже старта"),
    "rep_core":  ("low", -100, 100, "доля повторов (строчных букв) вокруг"),
    "rep_here":  ("low", 0, 0, "сам старт внутри повтора"),
}
DESCRIPTIONS = {
    "inr_pwm": "похожесть на Initiator: сумма log2(частота у стартов / фон) по буквам −5..+5",
    "inr_yr":  "классический Inr: пиримидин (C/T) на −1 и пурин (A/G) на 0",
    "div_inr": "дивергентный старт: лучший inr_pwm на противоположной цепи в 50–300 нт выше",
    "gc_1k": "доля G+C в ±1000 нт (широкий контекст)",
    "inr_vs_3": "inr_pwm минус лучший inr_pwm в ±3 нт (0 = лучшая позиция среди ближайших соседей)",
    "inr_vs_10": "то же в ±10 нт",
    "inr_vs_25": "то же в ±25 нт",
    "cpg_1k": "CpG наблюдаемое/ожидаемое в ±1000 нт",
    "rep_1k": "доля повторов в ±1000 нт",
    "polyAT_1k": "участки AAAAAA/TTTTTT в ±1000 нт",
    "splice_far": "доноры сплайсинга GTRAGT в +300..+2000 (дальше в теле гена)",
    **{k: v[4] for k, v in MOTIFS.items()},
    **{k: v[3] for k, v in COMPOSITION.items()},
}
GROUPS = {"сам старт": ["inr_pwm", "inr_yr"], "дивергентность": ["div_inr"],
          "мотивы промотора": ["tata", "ccaat", "ebox", "cre"], "гены тРНК (Pol III)": ["trna_abox", "trna_bbox"],
          "структура гена": ["splice_down"], "поли-A/T": ["polyAT_core", "polyAT_up"],
          "состав": ["gc_core", "gc_up", "gc_down", "cpg_core", "cpg_down"], "повторы": ["rep_core", "rep_here"],
          "локальный контраст": ["inr_vs_3", "inr_vs_10", "inr_vs_25"],
          "широкий контекст": ["gc_1k", "cpg_1k", "rep_1k", "polyAT_1k", "splice_far"]}
PWM_A, PWM_B = -5, 5


def oriented(seq, low, strand):
    """Последовательность (и маска повторов) по ходу транскрипции данной цепи."""
    if strand == "+":
        return seq, low
    return COMP[seq[::-1]], low[::-1]


def to_index(pos, L, strand):
    """Позиция генома -> индекс в ориентированной последовательности."""
    return pos if strand == "+" else L - 1 - pos


def motif_hits(o, motif, mismatches=0):
    """hit[j] = 1, если мотив (IUPAC) начинается в индексе j с не более чем `mismatches` несовпадениями."""
    m = len(motif); L = len(o)
    score = np.zeros(L, np.int8)
    for k, ch in enumerate(motif):
        ok = np.zeros(5, bool); ok[["ACGT".index(x) for x in IUPAC[ch]]] = True
        score[:L - k] += ok[o[k:]]
    hit = score >= m - mismatches
    hit[L - m + 1:] = False
    return hit


def window_sum(track, idx, a, b):
    """Сумма track[i+a .. i+b] для каждого i из idx (за краями контига считаем нули).
    Через накопленную сумму: сумма в окне = c[конец] - c[начало], поэтому длина окна на скорость не влияет."""
    c = np.zeros(len(track) + 1, np.int32 if track.dtype == bool else np.float64)
    np.cumsum(track, dtype=c.dtype, out=c[1:])
    lo = np.clip(idx + a, 0, len(track)); hi = np.clip(idx + b + 1, 0, len(track))
    return c[hi] - c[lo]


def window_max(track, idx, a, b, fill=0.0):
    """Максимум track[i+a .. i+b] для каждого i из idx; позиции за краями контига считаются равными fill."""
    n = b - a + 1
    pad = np.concatenate([np.full(n, fill, track.dtype), track, np.full(n, fill, track.dtype)])
    cen = maximum_filter1d(pad, n, mode="nearest")            # окно с центром: cen[k] = max(pad[k - n//2 : k - n//2 + n])
    j = np.clip(idx + a + n, 0, len(pad) - n)                 # нужно max(pad[j : j+n]) = cen[j + n//2]
    return cen[j + n // 2].astype(np.float32)


def pwm_track(o, pwm):
    """Счёт PWM (матрица 5 x W, смещения PWM_A..PWM_B) для каждого индекса как позиции старта."""
    L = len(o); s = np.zeros(L, np.float32)
    for k, off in enumerate(range(PWM_A, PWM_B + 1)):
        v = pwm[:, k].astype(np.float32)[o]
        if off >= 0: s[:L - off] += v[off:]
        else:        s[-off:] += v[:L + off]
    return s


def fit_pwm(pos_windows, bg, pseudo=1.0):
    """PWM из окон (n x W, коды 0..4) вокруг стартов: log2(частота буквы в колонке / фоновая частота). N даёт 0."""
    cnt = np.stack([(pos_windows == b).sum(0) for b in range(4)]).astype(float)
    f = (cnt + pseudo) / (cnt.sum(0) + 4 * pseudo)
    pwm = np.zeros((5, pos_windows.shape[1])); pwm[:4] = np.log2(f / np.asarray(bg)[:, None])
    return pwm


# Шаг 7: известные элементы промотора как непрерывный счёт (лучший сайт в позиционном окне, без порога)
CONT_ELEMENTS = {"tata": ("TATAWAWR", -40, -20), "dpe": ("RGWYV", 26, 34), "mte": ("CSARCSSAAC", 16, 29),
                 "gcbox": ("GGGCGG", -150, -30), "ccaat": ("CCAAT", -150, -40), "ebox": ("CACGTG", -100, 20)}


def consensus_pwm(cons):
    """Матрица счёта (длина x 4) из IUPAC-консенсуса: разрешённая буква +log2(1/(k*0.25)), остальные -3."""
    w = np.full((len(cons), 4), -3.0, np.float32)
    for k, ch in enumerate(cons):
        for b in IUPAC[ch]: w[k, "ACGT".index(b)] = 2.0 - np.log2(len(IUPAC[ch]))
    return w


def cont_names(params):
    names = []
    for m in params.get("tf", []):
        names += [f"tfmax_{m['name']}{suffix}" for suffix in m["windows"] if suffix != "_1k"]
    names += [f"cont_{n}" for n in CONT_ELEMENTS]
    return names + ["regime_inr_dens50", "regime_inr_dens250", "regime_inr_max50", "regime_cpg_dens100", "regime_gc_updown"]


def strand_features(o, lw, i, inr_here, inr_other, params):
    """Признаки для индексов i ориентированной последовательности o одной цепи; по одному признаку за раз
    (генератор пар (имя, значения)), чтобы не держать в памяти все дорожки сразу.
    inr_here / inr_other: дорожки inr_pwm этой и противоположной цепи (нужны для дивергентного старта)."""
    from motifs import score_track
    L = len(o)
    # --- сам старт ---
    yield "inr_pwm", inr_here[i]
    v = np.zeros(len(i), np.float32); ok = i >= 1
    v[ok] = ((o[i[ok] - 1] == 1) | (o[i[ok] - 1] == 3)) & ((o[i[ok]] == 0) | (o[i[ok]] == 2))
    yield "inr_yr", v
    # --- дивергентный старт на другой цепи: индекс той же позиции генома в другой ориентации, окно +50..+300 ---
    yield "div_inr", window_max(inr_other, L - 1 - i, 50, 300, fill=-50.0)
    # --- локальный контраст (шаг 3б): насколько эта позиция лучше соседей по Initiator ---
    if params.get("local"):
        for h in (3, 10, 25):
            yield f"inr_vs_{h}", inr_here[i] - window_max(inr_here, i, -h, h, fill=-50.0)
    # --- мотивы: число вхождений начала мотива в окне ---
    hits = {}                                                     # одна дорожка на мотив, даже если окон несколько
    for name, (m, mm, a, b, _) in MOTIFS.items():
        if (m, mm) not in hits:
            hits[m, mm] = (motif_hits(o, "AAAAAA") | motif_hits(o, "TTTTTT")) if m is None else motif_hits(o, m, mm)
        yield name, window_sum(hits[m, mm], i, a, b)
    del hits
    # --- состав ---
    valid = o < 4
    gc = (o == 1) | (o == 2)
    cg = np.zeros(L, bool); cg[:-1] = (o[:-1] == 1) & (o[1:] == 2)              # динуклеотид CpG начинается здесь
    for name, (kind, a, b, _) in COMPOSITION.items():
        if kind == "gc":
            yield name, window_sum(gc, i, a, b) / np.maximum(window_sum(valid, i, a, b), 1)
        elif kind == "low":
            yield name, window_sum(lw, i, a, b) / (b - a + 1)
        else:                                                     # CpG наблюдаемое / ожидаемое = n(CG) * n / (n(C) * n(G))
            n = np.maximum(window_sum(valid, i, a, b), 1)
            nc, ng = window_sum(o == 1, i, a, b), window_sum(o == 2, i, a, b)
            yield name, window_sum(cg, i, a, b) * n / np.maximum(nc * ng, 1)
    # --- k-меры, найденные по данным (шаг 2, раздел 3) ---
    for name, a, b in params.get("kmers", []):
        yield f"kmer_{name}", window_sum(motif_hits(o, name), i, a, b)
    # --- мотивы факторов транскрипции из JASPAR/HOCOMOCO (шаг 3): число сайтов со счётом >= порога в окне ---
    for m in params.get("tf", []):
        sc = score_track(o, np.asarray(m["pwm"], np.float32)); hit = sc >= m["thr"]
        for suffix, (a, b) in m["windows"].items():                 # один мотив может давать признаки в нескольких окнах
            yield f"tf_{m['name']}{suffix}", window_sum(hit, i, a, b)
            if params.get("cont") and suffix != "_1k":              # шаг 7: непрерывный счёт (без порога) — лучший сайт в окне
                yield f"tfmax_{m['name']}{suffix}", window_max(sc, i, a, b, fill=-50.0)
        del hit, sc
    if params.get("cont"):                                          # шаг 7
        for name, (cons, a, b) in CONT_ELEMENTS.items():
            yield f"cont_{name}", window_max(score_track(o, consensus_pwm(cons)), i, a, b, fill=-50.0)
        q = np.quantile(inr_here[::37], 0.99)                       # режим промотора: плотность Inr-подобных позиций вокруг
        strong = inr_here >= q
        yield "regime_inr_dens50", window_sum(strong, i, -50, 50)
        yield "regime_inr_dens250", window_sum(strong, i, -250, 250)
        yield "regime_inr_max50", window_max(inr_here, i, -50, 50, fill=-50.0)
        yield "regime_cpg_dens100", window_sum(cg, i, -100, 100)
        yield "regime_gc_updown", (window_sum(gc, i, -200, -1) - window_sum(gc, i, 1, 200)) / 200.0
    # --- широкий контекст ±1000 нт (шаг 3) ---
    if params.get("wide"):
        n = np.maximum(window_sum(valid, i, -1000, 1000), 1)
        yield "gc_1k", window_sum(gc, i, -1000, 1000) / n
        yield "cpg_1k", window_sum(cg, i, -1000, 1000) * n / np.maximum(window_sum(o == 1, i, -1000, 1000) * window_sum(o == 2, i, -1000, 1000), 1)
        yield "rep_1k", window_sum(lw, i, -1000, 1000) / 2001
        yield "polyAT_1k", window_sum(motif_hits(o, "AAAAAA") | motif_hits(o, "TTTTTT"), i, -1000, 1000)
        yield "splice_far", window_sum(motif_hits(o, "GTRAGT"), i, 300, 2000)


def contig_features(seq, low, positions, strands, params, only=None):
    """Все признаки для позиций одного контига. positions: позиции генома, strands: массив '+'/'-'.
    params: {'pwm': 5 x W, 'kmers': [(имя, a, b), ...], 'tf': [...]} — то, что выучено на fit-части.
    only: если задан список имён, возвращаем только эти признаки (остальные всё равно считаются, это просто фильтр)."""
    L = len(seq)
    out = {}
    inr = {st: pwm_track(oriented(seq, low, st)[0], params["pwm"]) for st in ("+", "-")}
    rows_all = np.arange(len(positions))
    for st in ("+", "-"):
        sel = rows_all[strands == st]
        if len(sel) == 0: continue
        o, lw = oriented(seq, low, st)
        i = to_index(positions[sel], L, st)
        for k, v in strand_features(o, lw, i, inr[st], inr["-" if st == "+" else "+"], params):
            if only is None or k in only:
                out.setdefault(k, np.zeros(len(positions), np.float32))[sel] = v
    return out


def feature_matrix(seq, low, positions, strands, params, names):
    """То же, что contig_features, но сразу в матрицу float32 (строки x names) в заданном порядке колонок:
    так экономнее по памяти для миллионов позиций (плотная валидация, тест)."""
    L = len(seq); col = {n: k for k, n in enumerate(names)}
    X = np.zeros((len(positions), len(names)), np.float32)
    inr = {st: pwm_track(oriented(seq, low, st)[0], params["pwm"]) for st in ("+", "-")}
    for st in ("+", "-"):
        sel = np.flatnonzero(strands == st)
        if len(sel) == 0: continue
        o, lw = oriented(seq, low, st)
        i = to_index(positions[sel], L, st)
        for k, v in strand_features(o, lw, i, inr[st], inr["-" if st == "+" else "+"], params):
            if k in col: X[sel, col[k]] = v
    return X


LOGO_A, LOGO_B = -50, 30


def logo_names(a=LOGO_A, b=LOGO_B):
    """Имена признаков-логотипа: буква X на смещении k от старта (по ходу транскрипции), например 'b-1_C', 'b+0_A'."""
    return [f"b{o:+d}_{'ACGT'[x]}" for o in range(a, b + 1) for x in range(4)]


def logo_onehot(seq, positions, strand, a=LOGO_A, b=LOGO_B):
    """Позиционный «логотип»: для каждой позиции one-hot буквы на смещениях a..b по ходу транскрипции (uint8, n x 4*(b-a+1)).
    Считается без дорожек, поэтому его легко собирать кусками прямо перед предсказанием."""
    off = np.arange(a, b + 1)
    g = positions[:, None] + (off if strand == "+" else -off)[None, :]
    ok = (g >= 0) & (g < len(seq))
    code = np.where(ok, seq[np.where(ok, g, 0)], 4)
    if strand == "-": code = COMP[code]
    out = np.zeros((len(positions), len(off), 4), np.uint8)
    for x in range(4): out[:, :, x] = code == x
    return out.reshape(len(positions), -1)


DINUC_A, DINUC_B = -5, 30


def dinuc_names(a=DINUC_A, b=DINUC_B):
    """Имена признаков «пара букв»: динуклеотид XY, начинающийся на смещении k (например 'd-1_CA' — C на −1 и A на 0)."""
    return [f"d{o:+d}_{'ACGT'[x // 4]}{'ACGT'[x % 4]}" for o in range(a, b + 1) for x in range(16)]


def dinuc_onehot(seq, positions, strand, a=DINUC_A, b=DINUC_B):
    """Пары соседних букв на смещениях a..b по ходу транскрипции, one-hot (uint8, n x 16*(b-a+1))."""
    off = np.arange(a, b + 2)
    g = positions[:, None] + (off if strand == "+" else -off)[None, :]
    ok = (g >= 0) & (g < len(seq))
    code = np.where(ok, seq[np.where(ok, g, 0)], 4)
    if strand == "-": code = COMP[code]
    c1, c2 = code[:, :-1], code[:, 1:]
    pair = np.where((c1 < 4) & (c2 < 4), c1 * 4 + c2, 16)
    out = np.zeros((len(positions), b - a + 1, 16), np.uint8)
    for x in range(16): out[:, :, x] = pair == x
    return out.reshape(len(positions), -1)
