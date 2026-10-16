"""Собирает epic_v3_bpnet.ipynb из ячеек ниже: python3 tools/build_v3_notebook.py"""
import json, os

CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r'''
# EPIC v3: мультивидовой BPNet + ансамбль с v2

**Что нового по сравнению с v2.**

1. **Архитектура BPNet / ProCapNet** (Avsec 2021, Cochran 2024): stem-свёртка → 9 остаточных dilated-свёрток (дилатации 2…512) → две головы:
   * **профиль**: логит на каждую позицию; softmax по центральной зоне окна = *где* внутри окна начинается транскрипция;
   * **счёт**: одно число на окно и цепь = log(1 + сколько всего прочтений в центральной зоне).
   Предсказание в позиции = счёт × доля профиля. Loss = кросс-энтропия профиля (multinomial) + MSE счёта.
2. **Мультивидовое обучение**: тело общее для всех видов, головы свои у каждого вида (выход свёртки/линейного слоя на `n_species` каналов).
   Учимся на целевом виде (по умолчанию клоп `Ofas_2.0`) и вспомогательных (моль, устрица, Nematostella): данные организаторов, правилами разрешено.
3. **Ансамбль с v2**: если рядом лежат предсказания v2 (`holdout_tune_lograte.npy`, `holdout_report_lograte.npy`, `test_lograte.npy`),
   подбираем на группе **tune** способ и вес смешивания, проверяем на **report**, сабмит делаем из смеси.

Группы tune/report, метрики (копия официального скорера), rank-power квантование, формат сабмита и продолжение с чекпоинта — как в v2.

**Запуск на Kaggle**: Accelerator = GPU T4 x2, Internet = On. Выполнить всё. Сабмит: `/kaggle/working/epic_v3/<вид>/<вид>.predictions.txt.gz`.
Если сессия кончилась посреди обучения — подключите выход прошлой версии ноутбука как Input и запустите снова: чекпоинты `*.pt` подхватятся.
''')

md("## 1. Окружение и настройки")
code(r'''
# 1.1 Библиотеки и устройство
import os, sys, threading, gzip, json, math, time, copy, random, glob, shutil, tarfile, hashlib, contextlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
from IPython.display import display

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    major, _ = torch.cuda.get_device_capability()
    # У BPNet после каждой свёртки BatchNorm, активации ограничены -> float16 на T4 безопасен (шаги с inf пропускает GradScaler).
    # EPIC_FP32=1 выключает смешанную точность совсем
    AMP_DTYPE = None if os.environ.get("EPIC_FP32") == "1" else (torch.bfloat16 if major >= 8 else torch.float16)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print("GPU:", torch.cuda.device_count(), "x", torch.cuda.get_device_name(), "| точность:", AMP_DTYPE or "float32")
else:
    AMP_DTYPE = None
    print("GPU нет: на CPU годится только проверка кода (EPIC_FAST_DEV=1)")
    if os.environ.get("EPIC_FAST_DEV") != "1" and os.environ.get("EPIC_REUSE_V3") != "1":
        raise RuntimeError("GPU не найден. Kaggle: Settings ноутбука -> Accelerator = GPU T4 x2 (нужен подтверждённый телефон)")

def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE is not None else contextlib.nullcontext()
''')

code(r'''
# 1.2 Все настраиваемые параметры в одном месте
TARGET = os.environ.get("EPIC_SPECIES", "Ofas_2.0")                       # вид, для которого делаем сабмит
AUX = [s for s in os.environ.get("EPIC_AUX", "ilPloInte3.2,cgigas_uk_roslin_v1,jaNemVect1.1").split(",") if s and s != TARGET]
SPECIES = [TARGET] + AUX                                                  # индекс 0 = целевой вид (его головы идут в сабмит)
FAST_DEV = os.environ.get("EPIC_FAST_DEV") == "1"
SESSION_T0 = time.time()
TAG = os.environ.get("EPIC_TAG", "v3bp")   # префикс файлов весов; новый TAG = новый прогон обучения
USE_GT_TEST = os.environ.get("EPIC_USE_GT_TEST", "1") == "1"   # тест вспомогательного вида с открытыми ответами (Nematostella) -> в обучение
# EPIC_CKPT_FROM: взять готовые веса, обученные для другого целевого вида (например Ofas_2.0, где моль и устрица были
# вспомогательными), и предсказать TARGET его собственными головами. Обучения при этом нет
CKPT_FROM = os.environ.get("EPIC_CKPT_FROM", TARGET)
HEAD = 0            # номер головы целевого вида; при CKPT_FROM != TARGET уточняется после загрузки весов (раздел 6.4)

ON_KAGGLE = os.path.isdir("/kaggle/working")
ON_COLAB = "google.colab" in sys.modules
if ON_KAGGLE:  default_data, default_out = "/tmp/epic", f"/kaggle/working/epic_v3/{TARGET}"
elif ON_COLAB: default_data, default_out = "/content/epic", f"/content/epic_v3/{TARGET}"
else:          default_data, default_out = os.path.expanduser("~/epic_data"), os.path.expanduser(f"~/epic_v3/{TARGET}")
DATA_DIR = os.environ.get("EPIC_DATA_DIR", default_data)
OUT_DIR = os.environ.get("EPIC_OUT_DIR", default_out)
os.makedirs(OUT_DIR, exist_ok=True)

# чекпоинты прошлых запусков (Kaggle: выход прошлой версии ноутбука, подключённый как Input)
IMPORT_CKPT_GLOB = os.environ.get("EPIC_IMPORT_CKPT", "/kaggle/input/**/*.pt" if ON_KAGGLE else "")
for src in sorted(glob.glob(IMPORT_CKPT_GLOB, recursive=True)) if IMPORT_CKPT_GLOB else []:
    dst = f"{OUT_DIR}/{os.path.basename(src)}"
    if os.path.basename(src).startswith(f"{CKPT_FROM}_{TAG}_") and not os.path.exists(dst):
        shutil.copy(src, dst); print("импортирован чекпоинт:", src)

CFG = dict(
    SPECIES=tuple(SPECIES),
    # --- данные ---
    WIN=4096,                 # окно ДНК на входе
    CROP=1024,                # края окна без loss и без предсказаний; рецептивное поле ~±1070 нт, поэтому центр видит полный контекст
    HOLDOUT_FRAC=0.05,        # как в v2: группы tune/report совпадут с v2 (нужно для ансамбля)
    USE_LOWERCASE=True,       # флаг строчной буквы (мягкая маска повторов) на входе
    POS_FRAC=0.5,             # доля окон вокруг положительной позиции
    TARGET_WEIGHT=0.5,        # доля окон целевого вида в батче; остальное поровну между AUX
    # --- модель ---
    CH=256, N_LAYERS=9, STEM_K=21, PROF_K=75,
    COUNT_WEIGHT=1.0,         # вес MSE головы счёта относительно кросс-энтропии профиля
    # --- обучение ---
    STEPS=30000, BATCH=32, LR=float(os.environ.get("EPIC_LR", "5e-4")),   # 1e-3 в прогоне 2026-10-08: seed0 разошёлся
    WEIGHT_DECAY=1e-2, WARMUP=0.05,
    DIVERGE_FRAC=0.5, EMA_DECAY=0.999,
    EVAL_EVERY=2000, EVAL_BP=4_000_000,
    N_SEEDS=int(os.environ.get("EPIC_N_SEEDS", "2")), CKPT_EVERY=1000,
    HO_MAX_BP=int(float(os.environ.get("EPIC_HO_MAX_BP", "0")) * 1e6) or None,   # оценка tune/report на выборке (Мб на группу): огромные хромосомы не влезают в память
    STOP_AFTER_HOURS=11.0 if ON_KAGGLE else None,
    # --- предсказание и постобработка ---
    TTA_SHIFTS=2, PRED_BATCH=32,
    KEEP_MODEL_FRAC=0.8,
    EXTRA_SUB_ALPHAS=(0.7, 0.8),  # дополнительно записать сабмиты с этими alpha (Spearman растёт с alpha, AUPRC немного падает)      # в ансамбль идут модели с AUPRC на tune >= этой доли от лучшей
    ALPHAS=(0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9), KEEP_FRAC=0.995,
    SELECT="lb",              # выбор смеси и alpha: "lb" — минимум произведения мест на текущем лидерборде вида (оценка по tune), "keep" — старое правило
    LB_TEAM="Mitochondrion", LB_SHORTLIST=6,
    LB_CALIB=(1.0, 1.0),      # множители tune -> лидерборд для (AUPRC, Spearman), если hold-out вида систематически легче/тяжелее теста   # своя команда (не считается соперником); сколько смесей из сетки проверять по alpha
    BLEND_WEIGHTS=tuple(np.round(np.arange(0, 1.01, 0.1), 2)),   # вес v3 в смеси с v2
)
if FAST_DEV:
    CFG.update(WIN=1024, CROP=256, CH=16, N_LAYERS=4, STEM_K=11, PROF_K=15, STEPS=40, BATCH=4,
               EVAL_EVERY=20, EVAL_BP=60_000, N_SEEDS=1, CKPT_EVERY=20, PRED_BATCH=8)
print("платформа:", "Kaggle" if ON_KAGGLE else "Colab" if ON_COLAB else "другая", "| целевой вид:", TARGET, "| вспомогательные:", AUX, "| FAST_DEV:", FAST_DEV)
print("данные:", DATA_DIR, "| результаты:", OUT_DIR)
display(pd.Series({k: str(v) for k, v in CFG.items()}, name="значение").to_frame())
''')

code(r'''
# 1.3 Архивы видов: из Input (/kaggle/input/**/<вид>.tgz, например датасет espernauto/epic-kittens) или с Zenodo
#     (doi:10.5281/zenodo.22285753); в обоих случаях MD5 сверяется с суммами Zenodo
ZENODO_MD5 = {"jaNemVect1.1": "733dccb6393a3c07b6e0d7ffaed3686f", "ASM119413v2": "108050274a2370f095ad508d62f5cde4",
              "Ofas_2.0": "c785a4b14346501d77e22ff645c511e9", "cgigas_uk_roslin_v1": "661e2c4d4e0378eddbb46bdc555110ef",
              "sSquSuc5.hap2": "3e4ae5b2f8f45b3a41d78b4c87eed5d2", "ilPloInte3.2": "b0317e9e686699089665616d3b87d3fa"}

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()

def download_dataset(species, data_dir):
    if glob.glob(f"{data_dir}/{species}*/for_participants"):
        print("уже распакован:", species); return
    os.makedirs(data_dir, exist_ok=True)
    local = sorted(glob.glob(f"/kaggle/input/**/{species}.tgz", recursive=True))
    if local:
        src = local[0]; print("архив из Input:", src)
    else:
        import requests
        rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
        f = [x for x in rec["files"] if x["key"] == f"{species}.tgz"][0]
        src = f"{data_dir}/{species}.tgz"
        with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(src, "wb") as fh:
            r.raise_for_status()
            for chunk in tqdm(r.iter_content(1 << 22), total=f["size"] // (1 << 22) + 1, desc=f"скачиваю {species}", unit="×4МБ"):
                fh.write(chunk)
    got = md5_file(src)
    assert got == ZENODO_MD5[species], f"MD5 архива {species} не совпал с Zenodo: {got}"
    with tarfile.open(src) as tf:
        try: tf.extractall(data_dir, filter="data")
        except TypeError: tf.extractall(data_dir)
    if not local: os.remove(src)

BASES = {}
for sp in SPECIES:
    download_dataset(sp, DATA_DIR)
    BASES[sp] = os.path.dirname(sorted(glob.glob(f"{DATA_DIR}/{sp}*/for_participants"))[0])
    print(sp, "->", BASES[sp])
''')

md(r'''
## 2. Данные всех видов

Для каждого вида: геном (коды 0–3 = ACGT, 4–7 = acgt, 8 = N), маска whitelist (train, а у целевого вида ещё и test — по шаблону),
метки (сумма реплик csRNA-seq по позициям-цепям). Код тот же, что в v2, только обёрнут в функцию вида.
''')
code(r'''
# 2.1 Загрузка
def read_bed(path, ncols):
    names = ["chrom", "start", "end", "name", "score", "strand"][:ncols]
    return pd.read_csv(path, sep="\t", header=None, usecols=range(ncols), names=names, dtype={"chrom": str, "strand": str})

LUT = np.full(256, 8, np.uint8)
for i, b in enumerate("ACGT"):
    LUT[ord(b)] = i; LUT[ord(b.lower())] = i + 4

def read_genome(path, keep):
    genome, name, chunks = {}, None, []
    def flush():
        if name in keep: genome[name] = LUT[np.frombuffer(b"".join(chunks), np.uint8)]
    with gzip.open(path, "rb") as fh:
        for line in tqdm(fh, desc=f"геном {os.path.basename(os.path.dirname(os.path.dirname(path)))}", unit=" строк", mininterval=5):
            if line.startswith(b">"):
                flush(); name = line[1:].split()[0].decode(); chunks = []
            elif name in keep:
                chunks.append(line.rstrip())
    flush()
    return genome

def read_counts_bed(path):
    d = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 4, 5], names=["chrom", "start", "end", "count", "strand"],
                    dtype={"chrom": str, "strand": str})
    span = (d.end - d.start).values
    first = np.repeat(np.cumsum(span) - span, span)
    return pd.DataFrame({"chrom": np.repeat(d.chrom.values, span),
                         "pos": np.repeat(d.start.values, span) + (np.arange(span.sum()) - first),
                         "strand": np.repeat((d.strand.values == "-").astype(np.int8), span),
                         "count": np.repeat(d["count"].values, span).astype(np.float32)})

def load_species(sp, with_test):
    fp = f"{BASES[sp]}/for_participants"
    wl = read_bed(f"{fp}/whitelist.train.bed.gz", 3)
    tpl = read_bed(f"{fp}/template.test.bed.gz", 6) if with_test else None
    # вспомогательный вид с открытыми ответами теста (Nematostella): тест тоже идёт в обучение (правилами разрешено)
    gt_path = f"{BASES[sp]}/ground_truth/csRNA-test.txt.gz"
    extra = None
    if not with_test and USE_GT_TEST and os.path.exists(gt_path):
        tg = read_bed(f"{fp}/template.test.bed.gz", 6)
        gt = pd.read_csv(gt_path, header=None, dtype=np.float32)[0].values
        off = np.r_[0, np.cumsum((tg.end - tg.start).values)]
        assert off[-1] == len(gt), f"{sp}: длина ответов теста не совпала с шаблоном"
        nz = np.flatnonzero(gt > 0); k = np.searchsorted(off, nz, side="right") - 1
        extra = pd.DataFrame({"chrom": tg.chrom.values[k], "pos": tg.start.values[k] + (nz - off[k]),
                              "strand": (tg.strand.values[k] == "-").astype(np.int8), "tot": gt[nz]})
        wl = pd.concat([wl, tg[["chrom", "start", "end"]].drop_duplicates()], ignore_index=True)
        print(f"{sp}: + тест с ответами: {tg.chrom.nunique()} контигов, {len(extra):,} положительных позиций-цепей")
    train_c = list(dict.fromkeys(wl.chrom))
    test_c = list(dict.fromkeys(tpl.chrom)) if with_test else []
    genome = read_genome(f"{BASES[sp]}/genome/genome.fa.gz", set(train_c) | set(test_c))
    missing = [c for c in train_c + test_c if c not in genome]
    assert not missing, f"{sp}: в fasta нет контигов {missing[:5]}"
    in_wl = {c: np.zeros(len(genome[c]), bool) for c in train_c + test_c}
    for c, s, e in zip(wl.chrom, wl.start, wl.end): in_wl[c][s:e] = True
    if with_test:
        for c, s, e in zip(tpl.chrom, tpl.start, tpl.end): in_wl[c][s:e] = True
    files = sorted(glob.glob(f"{fp}/csRNA-*r[0-9]-train.bed.gz"))
    assert files, f"{sp}: не нашёл csRNA-*-train.bed.gz"
    lab = None
    for i, path in enumerate(files):
        r = read_counts_bed(path).rename(columns={"count": f"r{i}"})
        lab = r if lab is None else lab.merge(r, on=["chrom", "pos", "strand"], how="outer")
    rep_cols = [f"r{i}" for i in range(len(files))]
    lab["tot"] = lab[rep_cols].fillna(0).sum(axis=1).astype(np.float32)
    if extra is not None: lab = pd.concat([lab[["chrom", "pos", "strand", "tot"]], extra], ignore_index=True)
    track = {}
    for c, d in lab.groupby("chrom", sort=False):
        d = d.sort_values(["pos", "strand"])
        track[c] = dict(pos=d.pos.values.astype(np.int64), strand=d.strand.values.astype(np.int64), tot=d.tot.values.astype(np.float32))
    return dict(name=sp, wl=wl, tpl=tpl, train=train_c, test=test_c, genome=genome, in_wl=in_wl, track=track, files=[os.path.basename(f) for f in files])

DS = {sp: load_species(sp, sp == TARGET) for sp in SPECIES}
T = DS[TARGET]

info = {}
for sp, d in DS.items():
    wl_mb = int((d["wl"].end - d["wl"].start).sum()) / 1e6
    npos = sum(len(t["pos"]) for t in d["track"].values())
    info[sp] = {"контигов train": len(d["train"]), "Мб whitelist train": round(wl_mb, 1), "положительных позиций-цепей": npos,
                "доля положительных": f"{npos / (2 * wl_mb * 1e6):.2e}", "файлы меток": ", ".join(d["files"])}
    if d["tpl"] is not None:
        info[sp]["позиций в сабмите"] = f"{int((d['tpl'].end - d['tpl'].start).sum()):,}"
display(pd.DataFrame(info).T)
''')

md(r'''
## 3. Отложенные контиги каждого вида: `tune` и `report`

Тот же жадный алгоритм и те же настройки, что в v2, поэтому у целевого вида группы совпадают с группами v2 (это проверяется при смешивании).
Группы откладываются **у каждого вида**, не только у целевого: тогда веса этого прогона можно честно проверить и на вспомогательном виде
(режим `EPIC_CKPT_FROM`). В прогоне 2026-10-08 вспомогательные виды учились на всех своих контигах, и report моли/устрицы был завышен.
''')
code(r'''
# 3.1 Группы tune и report для каждого вида (алгоритм v2)
def make_groups(ds, frac):
    wl_bp = (ds["wl"].assign(bp=ds["wl"].end - ds["wl"].start).groupby("chrom", sort=False).bp.sum())
    pos_n = pd.Series({c: len(ds["track"][c]["pos"]) if c in ds["track"] else 0 for c in wl_bp.index})
    rate = pos_n.sum() / (2 * wl_bp.sum())
    def pick_group(avail, target_bp):
        sel, s_bp, s_pos = [], 0, 0
        while s_bp < target_bp and avail:
            best = min(avail, key=lambda c: abs((s_pos + pos_n[c]) / (2 * (s_bp + wl_bp[c])) - rate))
            sel.append(best); s_bp += wl_bp[best]; s_pos += pos_n[best]; avail.remove(best)
        return sel
    min_bp = max(min(100_000, int(wl_bp.max())), int(wl_bp.sum()) // 1000)
    avail = [c for c in wl_bp.index if wl_bp[c] >= min_bp]
    return {g: pick_group(avail, frac * wl_bp.sum()) for g in ("tune", "report")}, wl_bp, pos_n

GROUPS_ALL, FIT = {}, {}
for sp in SPECIES:
    GROUPS_ALL[sp], wl_bp_sp, _ = make_groups(DS[sp], CFG["HOLDOUT_FRAC"])
    held = set(GROUPS_ALL[sp]["tune"]) | set(GROUPS_ALL[sp]["report"])
    FIT[sp] = [c for c in DS[sp]["train"] if c not in held]
GROUPS, wl_bp, pos_n = make_groups(T, CFG["HOLDOUT_FRAC"])
assert GROUPS == GROUPS_ALL[TARGET]
json.dump(GROUPS, open(f"{OUT_DIR}/holdout_groups.json", "w"))
json.dump(GROUPS_ALL, open(f"{OUT_DIR}/holdout_groups_all.json", "w"))
display(pd.DataFrame({g: {"контигов": len(cs), "Мб whitelist": round(wl_bp[cs].sum() / 1e6, 1),
                          "доля положительных": f"{pos_n[cs].sum() / (2 * wl_bp[cs].sum()):.2e}"}
                      for g, cs in [("обучение", FIT[TARGET]), ("tune", GROUPS["tune"]), ("report", GROUPS["report"])]}).T)
if CKPT_FROM != TARGET:
    src_groups = [p for p in glob.glob(f"/kaggle/input/**/{CKPT_FROM}/holdout_groups_all.json", recursive=True)]
    if src_groups and TARGET in (sg := json.load(open(src_groups[0]))):
        assert sg[TARGET] == GROUPS, f"группы {TARGET} не совпадают с отложенными в прогоне {CKPT_FROM}"
        print(f"группы tune/report {TARGET} совпадают с отложенными в прогоне {CKPT_FROM}: report честный")
    else:
        print(f"ВНИМАНИЕ: веса из прогона {CKPT_FROM} без holdout_groups_all.json (прогоны до 2026-10-09): "
              f"{TARGET} учился на своих tune/report, метрики report завышены")
''')

md(r'''
## 4. Обучающие окна

Каждое окно батча сначала получает вид (целевой с вероятностью `TARGET_WEIGHT`, остальные поровну), затем строится как в v2:
с вероятностью `POS_FRAC` вокруг случайной положительной позиции (она попадает в случайное место центральной зоны), иначе равномерно по whitelist.
Цель окна — счёт на обеих цепях `(2, WIN)`, маска — whitelist.
''')
code(r'''
# 4.1 Окна и сэмплер
def window_codes(ds, c, start, L):
    seq = ds["genome"][c]; out = np.full(L, 8, np.uint8)
    a, b = max(start, 0), min(start + L, len(seq))
    if b > a: out[a - start:b - start] = seq[a:b]
    return out

def window_mask(ds, c, start, L):
    m = np.zeros(L, np.float32); a, b = max(start, 0), min(start + L, len(ds["genome"][c]))
    if c in ds["in_wl"] and b > a: m[a - start:b - start] = ds["in_wl"][c][a:b]
    return m

def window_targets(ds, c, start, L):
    y = np.zeros((2, L), np.float32); t = ds["track"].get(c)
    if t is not None:
        i0, i1 = np.searchsorted(t["pos"], [start, start + L])
        y[t["strand"][i0:i1], t["pos"][i0:i1] - start] = t["tot"][i0:i1]
    return y

class SpeciesSampler:
    def __init__(self, ds, contigs, cfg, rng):
        self.ds, self.cfg, self.rng = ds, cfg, rng
        iv = ds["wl"][ds["wl"].chrom.isin(set(contigs))]
        self.iv_c, self.iv_s = iv.chrom.values, iv.start.values
        self.iv_len = (iv.end - iv.start).values; self.iv_cum = np.cumsum(self.iv_len)
        have = [c for c in contigs if c in ds["track"]]
        self.pos_c = np.concatenate([np.full(len(ds["track"][c]["pos"]), i) for i, c in enumerate(have)])
        self.pos_p = np.concatenate([ds["track"][c]["pos"] for c in have]); self.pos_names = have

    def centre(self):
        if self.rng.random() < self.cfg["POS_FRAC"]:
            j = self.rng.integers(len(self.pos_p)); return self.pos_names[self.pos_c[j]], int(self.pos_p[j])
        u = self.rng.integers(self.iv_cum[-1]); k = np.searchsorted(self.iv_cum, u, side="right")
        return self.iv_c[k], int(self.iv_s[k] + u - (self.iv_cum[k] - self.iv_len[k]))

class MultiSampler:
    def __init__(self, cfg, seed):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)       # один генератор на всё: его состояние сохраняется в чекпоинт
        self.samplers = [SpeciesSampler(DS[sp], FIT[sp], cfg, self.rng) for sp in SPECIES]
        w = np.array([cfg["TARGET_WEIGHT"]] + [(1 - cfg["TARGET_WEIGHT"]) / max(len(AUX), 1)] * len(AUX)) if AUX else np.ones(1)
        self.p = w / w.sum()

    def batch(self, n):
        W, C = self.cfg["WIN"], self.cfg["CROP"]
        sp = self.rng.choice(len(SPECIES), size=n, p=self.p)
        codes, ys, masks = [], [], []
        for k in sp:
            smp = self.samplers[k]; c, p = smp.centre()
            start = p - C - int(self.rng.integers(W - 2 * C))
            codes.append(window_codes(smp.ds, c, start, W)); ys.append(window_targets(smp.ds, c, start, W)); masks.append(window_mask(smp.ds, c, start, W))
        return (torch.from_numpy(np.stack(codes)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(masks)),
                torch.from_numpy(sp.astype(np.int64)))

_s = MultiSampler(CFG, 0); _, _y, _m, _sp = _s.batch(256)
print("окон каждого вида в 256:", dict(zip(SPECIES, np.bincount(_sp.numpy(), minlength=len(SPECIES)))),
      "| окон с прочтениями в центре:", int((_y[:, :, CFG["CROP"]:-CFG["CROP"]].sum((1, 2)) > 0).sum()))
del _s, _y, _m, _sp
''')

md(r'''
## 5. Модель BPNetMS

```
вход (B, 6, WIN): one-hot ACGT + флаг строчной буквы + флаг N
  → stem: conv(STEM_K) → BN → ReLU                                (CH каналов)
  → N_LAYERS остаточных блоков: h + ReLU(BN(conv3, дилатация 2^i))  i = 1..N_LAYERS
  ├→ голова «профиль»: conv(PROF_K) → n_species каналов   (B, S, WIN)  логиты по позициям
  └→ голова «счёт»: среднее по центральной зоне → Linear → n_species   (B, S)  log(1 + сумма прочтений)
```
Вид окна выбирает свой канал обеих голов. Минус-цепь = та же сеть на обратно-комплементарном окне (выход развёрнут), как в v2.

**Предсказание в позиции** (log ожидаемого счёта) = log(expm1(счёт)) + log_softmax(профиль по позициям whitelist центральной зоны).
''')
code(r'''
# 5.1 Архитектура
def code_table(device):
    t = torch.zeros(9, 6, device=device)
    for b in range(4):
        t[b, b] = 1; t[b + 4, b] = 1; t[b + 4, 4] = 1
    t[8, 5] = 1
    return t

_TABLES = {}
RC_PERM = [3, 2, 1, 0, 4, 5]

def encode(codes, use_lowercase=True):
    if codes.device not in _TABLES: _TABLES[codes.device] = code_table(codes.device)
    x = _TABLES[codes.device][codes.long()].permute(0, 2, 1)
    if not use_lowercase: x = x * torch.tensor([1, 1, 1, 1, 0, 1.], device=x.device).view(1, 6, 1)
    return x

def rc(x):
    return x[:, RC_PERM].flip(-1)

class DilatedBlock(nn.Module):
    def __init__(self, ch, d):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=d, dilation=d); self.bn = nn.BatchNorm1d(ch)
    def forward(self, x):
        return x + F.relu(self.bn(self.conv(x)))

class BPNetMS(nn.Module):
    def __init__(self, n_species, ch=256, n_layers=9, stem_k=21, prof_k=75, in_ch=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, ch, stem_k, padding=stem_k // 2), nn.BatchNorm1d(ch), nn.ReLU())
        self.body = nn.Sequential(*[DilatedBlock(ch, 2 ** i) for i in range(1, n_layers + 1)])
        self.prof = nn.Conv1d(ch, n_species, prof_k, padding=prof_k // 2)
        self.count = nn.Linear(ch, n_species)

    def forward(self, x, crop):                                     # (B, 6, L) -> профиль (B, S, L), счёт (B, S)
        h = self.body(self.stem(x))
        return self.prof(h), self.count(h[..., crop:h.shape[-1] - crop].mean(-1))

def forward_both(net, x, sp, crop):
    # обе цепи одним батчем 2B; sp (B,) — индекс вида каждого окна -> профиль (B, 2, L), счёт (B, 2)
    B = x.shape[0]
    prof, cnt = net(torch.cat([x, rc(x)]), crop)
    rows, idx = torch.arange(2 * B, device=x.device), torch.cat([sp, sp])
    prof, cnt = prof[rows, idx], cnt[rows, idx]
    return torch.stack([prof[:B], prof[B:].flip(-1)], 1), torch.stack([cnt[:B], cnt[B:]], 1)

def build_model(cfg):
    return BPNetMS(len(cfg["SPECIES"]), cfg["CH"], cfg["N_LAYERS"], cfg["STEM_K"], cfg["PROF_K"])

def receptive_field(cfg):
    return (cfg["STEM_K"] - 1) + 2 * sum(2 ** i for i in range(1, cfg["N_LAYERS"] + 1)) + (cfg["PROF_K"] - 1) + 1

_net = build_model(CFG)
print(f"параметров: {sum(p.numel() for p in _net.parameters()):,}; рецептивное поле {receptive_field(CFG)} нт "
      f"(±{receptive_field(CFG) // 2}), CROP = {CFG['CROP']}, центральная зона окна {CFG['WIN'] - 2 * CFG['CROP']} нт")
with torch.no_grad():
    _p, _c = forward_both(_net.eval(), encode(torch.randint(0, 9, (2, CFG["WIN"]))), torch.tensor([0, len(SPECIES) - 1]), CFG["CROP"])
print("выходы на 2 окнах:", tuple(_p.shape), tuple(_c.shape))
del _net, _p, _c
''')

md("## 6. Метрики, loss и обучение")
code(r'''
# 6.1 Метрики (копия официального скорера) и rank-power квантование, как в v2
SCALE = 100_000

def quantise(pred):
    return np.rint(np.clip(pred, 0.0, 1.0) * SCALE).astype(np.int32)

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
    if mask is not None: pred01, gt = pred01[mask], gt[mask]
    lv, y = quantise(pred01), gt > 0
    return {"AUPRC": prauc_from_levels(lv, y), "Spearman": spearman_dense(lv[y], gt[y])}

def raw_score(pred, gt, mask=None):
    if mask is not None: pred, gt = pred[mask], gt[mask]
    y = gt > 0
    return {"AUPRC": float(average_precision_score(y, pred)), "Spearman": spearman_dense(pred[y], gt[y])}

def rank_power(score, alpha, nmask=None, n_ref=None):
    # место r (1 = сильнейшее) -> r^(-alpha), 5 знаков; N -> 0. float64: ранги смеси не должны слипаться.
    # n_ref: размер набора, под который подбираем alpha (не-N позиции теста). Ранги растягиваются в n_ref / n раз,
    # чтобы округление до 5 знаков вело себя как в сабмите: иначе на hold-out (68 млн) нулей намного меньше, чем в тесте (328 млн)
    # память: на пике два массива по 8 байт на позицию (у осьминога тест 417 млн позиций)
    x = np.negative(np.asarray(score, dtype=np.float64))
    if nmask is not None: x[nmask] = np.inf
    order = np.argsort(x, kind="stable"); del x
    n = len(order)
    r = np.empty(n, np.float64); r[order] = np.arange(1, n + 1, dtype=np.float64); del order
    n_valid = n - (int(nmask.sum()) if nmask is not None else 0)
    if alpha == 0: r -= 1; r /= -n; r += 1
    else:
        if n_ref: r *= n_ref / n_valid
        np.power(r, -alpha, out=r)
    np.clip(r, 0, 1, out=r); r *= SCALE; np.rint(r, out=r); r /= SCALE
    s = r.astype(np.float32); del r
    if nmask is not None: s[nmask] = 0.0
    return s
''')

code(r'''
# 6.2 Loss BPNet, переход к log-rate, EMA, быстрая проверка на tune
def bpnet_loss(prof, cnt, y, mask, cfg):
    # prof, y: (B, 2, L); cnt: (B, 2); mask: (B, L). Всё — только по центральной зоне и whitelist
    C = cfg["CROP"]
    p, yy = prof[..., C:-C].float(), y[..., C:-C]
    m = (mask[:, None, C:-C] > 0).expand_as(yy)
    logp = torch.log_softmax(p.masked_fill(~m, -1e4), -1)
    tot = (yy * m).sum(-1)                                           # (B, 2) прочтений в окне на цепи
    ce = -(yy * m * logp).sum(-1) / tot.clamp(min=1.0)               # кросс-энтропия к наблюдаемому распределению
    # Оригинальный multinomial NLL = tot * ce: окна с сотнями тысяч прочтений забивали бы всё обучение.
    # Берём вес log(1 + tot): окна без прочтений не учат профиль, сильные важнее слабых, но без перекоса
    w = torch.log1p(tot)
    prof_loss = (w * ce).sum() / w.sum().clamp(min=1e-6)
    cnt_loss = F.mse_loss(cnt.float(), torch.log1p(tot))
    return prof_loss + cfg["COUNT_WEIGHT"] * cnt_loss, (prof_loss, cnt_loss)

def to_lograte(prof, cnt, mask, crop):
    # -> (B, 2, L - 2*crop): log ожидаемого счёта в позиции = log(сумма) + log(доля профиля)
    C = crop
    p = prof[..., C:-C].float(); m = (mask[:, None, C:-C] > 0).expand_as(p)
    logp = torch.log_softmax(p.masked_fill(~m, -1e4), -1)
    return logp + torch.log(torch.expm1(cnt.float().clamp(min=1e-4, max=20)))[..., None]

class EMA:
    def __init__(self, model, decay):
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_lerp_(list(self.model.parameters()), [p.detach() for p in model.parameters()], 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()): e.copy_(b)

@torch.no_grad()
def predict_windows(models, codes, masks, cfg):
    # средний по моделям log-rate целевого вида (индекс 0): (n, 2, WIN - 2*CROP)
    out = []
    for i in range(0, len(codes), cfg["PRED_BATCH"]):
        c, mk = torch.from_numpy(codes[i:i + cfg["PRED_BATCH"]]), torch.from_numpy(masks[i:i + cfg["PRED_BATCH"]])
        res = []
        for net in models:
            net.eval(); dev = next(net.parameters()).device
            sp = torch.full((len(c),), HEAD, dtype=torch.long, device=dev)
            with autocast(): prof, cnt = forward_both(net, encode(c.to(dev), cfg["USE_LOWERCASE"]), sp, cfg["CROP"])
            res.append(to_lograte(prof, cnt, mk.to(dev), cfg["CROP"]).cpu())
        out.append((sum(res) / len(res)).numpy())
    return np.concatenate(out)

def make_eval_set(contigs, cfg, seed=0):
    W, C = cfg["WIN"], cfg["CROP"]; step = W - 2 * C; rng = np.random.default_rng(seed)
    starts = [(c, s) for c in contigs for s in range(-C, len(T["genome"][c]) - C, step) if T["in_wl"][c][max(s + C, 0):s + C + step].any()]
    starts = [starts[i] for i in rng.permutation(len(starts))[: max(1, cfg["EVAL_BP"] // step)]]
    codes = np.stack([window_codes(T, c, s, W) for c, s in starts])
    masks = np.stack([window_mask(T, c, s, W) for c, s in starts])
    y = np.stack([window_targets(T, c, s, W) for c, s in starts])[:, :, C:W - C]
    m = (masks[:, None, C:W - C] > 0) & (codes[:, None, C:W - C] != 8)
    return dict(codes=codes, masks=masks, y=y, mask=np.broadcast_to(m, y.shape))

def quick_eval(models, ev, cfg):
    lr = predict_windows(models, ev["codes"], ev["masks"], cfg)
    return raw_score(lr[ev["mask"]], ev["y"][ev["mask"]])

EVAL_SET = make_eval_set(GROUPS["tune"], CFG)
print(f"быстрая проверка: {len(EVAL_SET['codes'])} окон, позиций {int(EVAL_SET['mask'].sum()):,}, положительных {int((EVAL_SET['y'][EVAL_SET['mask']] > 0).sum()):,}")
''')

code(r'''
# 6.3 Обучение одной модели (с продолжением с любого шага), схема как в v2
class TrainingPaused(Exception):
    pass

SESSION_KEYS = ("STOP_AFTER_HOURS", "CKPT_EVERY", "EVAL_EVERY", "PRED_BATCH")

def ckpt_path(seed):
    return f"{OUT_DIR}/{CKPT_FROM}_{TAG}_seed{seed}.pt"

INIT_LOCK = threading.Lock()

def train_model(cfg, seed, gpu=None):
    if gpu is not None: torch.cuda.set_device(gpu)
    final = ckpt_path(seed); last = final.replace(".pt", ".last.pt")
    if os.path.exists(final):
        ck = torch.load(final, map_location=DEVICE, weights_only=False)
        net = build_model(ck["cfg"]).to(DEVICE); net.load_state_dict(ck["model"]); net.eval()
        print("загружена готовая модель", final); return net, ck["hist"], ck["evals"]

    assert CKPT_FROM == TARGET, f"нет готовых весов {final}: при EPIC_CKPT_FROM обучение не запускается"
    assert os.environ.get("EPIC_REUSE_V3") != "1", f"нет готовых весов {final}: в режиме EPIC_REUSE_V3 (CPU) обучение не запускается"
    state = torch.load(last, map_location=DEVICE, weights_only=False) if os.path.exists(last) else None
    if state is not None:
        diff = {k: (state["cfg"][k], cfg[k]) for k in cfg if k not in SESSION_KEYS and k in state["cfg"] and state["cfg"][k] != cfg[k]}
        if diff: print("в чекпоинте другие настройки (старое, новое), продолжаю со старыми:", diff)
        cfg = {**state["cfg"], **{k: cfg[k] for k in SESSION_KEYS if k in cfg}}

    with INIT_LOCK:
        random.seed(seed); np.random.seed(seed); torch.default_generator.manual_seed(seed)
        if DEVICE == "cuda": torch.cuda.manual_seed(seed)
        net = build_model(cfg).to(DEVICE)
    ema = EMA(net, cfg["EMA_DECAY"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["LR"], total_steps=cfg["STEPS"], pct_start=cfg["WARMUP"])
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    sampler = MultiSampler(cfg, seed)
    hist, evals, best, best_state, bad, start = [], [], -1.0, None, 0, 1
    lr_scale, rollbacks, run_loss = 1.0, 0, None
    rand_level = float((EVAL_SET["y"][EVAL_SET["mask"]] > 0).mean())

    if state is not None:
        net.load_state_dict(state["model"]); ema.model.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        if scaler.is_enabled() and state["scaler"]: scaler.load_state_dict(state["scaler"])
        sampler.rng.bit_generator.state = state["sampler_rng"]
        with INIT_LOCK: torch.set_rng_state(state["torch_rng"])
        hist, evals, best, best_state, bad = state["hist"], state["evals"], state["best"], state["best_state"], state["bad"]
        lr_scale, rollbacks = state.get("lr_scale", 1.0), state.get("rollbacks", 0)
        start = state["step"] + 1
        print(f"продолжаю {os.path.basename(last)} с шага {start} из {cfg['STEPS']}")
        del state

    def save_last(step):
        tmp = last + ".tmp"
        torch.save(dict(step=step, cfg=dict(cfg), seed=seed, model=net.state_dict(), ema=ema.model.state_dict(),
                        opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        sampler_rng=sampler.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        hist=hist, evals=evals, best=best, best_state=best_state, bad=bad, lr_scale=lr_scale, rollbacks=rollbacks), tmp)
        os.replace(tmp, last)

    pbar = tqdm(range(start, cfg["STEPS"] + 1), initial=start - 1, total=cfg["STEPS"], desc=f"seed {seed}" + (f" (GPU {gpu})" if gpu is not None else ""),
                unit="шаг", mininterval=5, position=gpu or 0, leave=True)
    for step in pbar:
        net.train()
        codes, y, mask, sp = (t.to(DEVICE, non_blocking=True) for t in sampler.batch(cfg["BATCH"]))
        with autocast():
            prof, cnt = forward_both(net, encode(codes, cfg["USE_LOWERCASE"]), sp, cfg["CROP"])
        loss, (lp, lc) = bpnet_loss(prof, cnt, y, mask, cfg)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if torch.isfinite(gn): scaler.step(opt); ema.update(net, step)
            else: bad += 1
            scaler.update()
        else:
            bad += 1; gn = torch.tensor(float("nan"))
        sched.step()
        if lr_scale != 1.0:
            for g in opt.param_groups: g["lr"] *= lr_scale
        if step % 10 == 0 or step == 1:
            lv = loss.item()
            if math.isfinite(lv): run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
            hist.append(dict(step=step, loss=lv, loss_prof=lp.item(), loss_cnt=lc.item(), lr=opt.param_groups[0]["lr"], grad_norm=float(gn), bad_steps=bad))
            pbar.set_postfix(loss=f"{run_loss:.4f}" if run_loss is not None else "nan", best_AUPRC=f"{best:.4f}", bad=bad)
        if step % cfg["EVAL_EVERY"] == 0 or step == cfg["STEPS"]:
            r_raw, r_ema = quick_eval([net], EVAL_SET, cfg), quick_eval([ema.model], EVAL_SET, cfg)
            evals.append(dict(step=step, AUPRC_raw=r_raw["AUPRC"], AUPRC_ema=r_ema["AUPRC"], Spearman_raw=r_raw["Spearman"], Spearman_ema=r_ema["Spearman"]))
            tqdm.write(f"[seed {seed}] шаг {step}: AUPRC tune {r_raw['AUPRC']:.4f} (EMA {r_ema['AUPRC']:.4f}), Spearman {r_raw['Spearman']:.3f} (EMA {r_ema['Spearman']:.3f})")
            if r_ema["AUPRC"] > best:
                best, best_state = r_ema["AUPRC"], copy.deepcopy(ema.model.state_dict())
            elif best > 2 * rand_level and r_ema["AUPRC"] < cfg["DIVERGE_FRAC"] * best:
                rollbacks += 1; lr_scale *= 0.5
                net.load_state_dict(best_state); ema.model.load_state_dict(best_state); opt.state.clear()
                tqdm.write(f"  ! [seed {seed}] AUPRC упал до {r_ema['AUPRC']:.4f}: откат к лучшим весам, LR x{lr_scale:g}")
                if rollbacks > 4:   # сид не учится: останавливаем его на лучших весах; отбор моделей (KEEP_MODEL_FRAC) решит, брать ли его
                    tqdm.write(f"  ! [seed {seed}] 5 откатов: обучение этого сида остановлено на шаге {step} с лучшими весами"); pbar.close(); break
        if step < cfg["STEPS"]:
            if step % cfg["CKPT_EVERY"] == 0: save_last(step)
            hours = cfg.get("STOP_AFTER_HOURS")
            if hours and time.time() - SESSION_T0 > hours * 3600:
                save_last(step); pbar.close()
                raise TrainingPaused(f"Время сессии вышло на шаге {step}; состояние в {last}. Подключите выход этой версии как Input и запустите снова.")

    if best_state is None: best_state = copy.deepcopy(ema.model.state_dict())
    ema.model.load_state_dict(best_state)
    torch.save(dict(model=best_state, cfg=dict(cfg), hist=hist, evals=evals, seed=seed), final)
    if os.path.exists(last): os.remove(last)
    print(f"сохранено {final}; лучший AUPRC tune (EMA) {best:.4f}; пропущено шагов {bad}")
    return ema.model.eval(), hist, evals
''')

code(r'''
# 6.4 Обучаем N_SEEDS моделей; на Kaggle T4 x2 — параллельно, по потоку на GPU
N_GPU = torch.cuda.device_count() if DEVICE == "cuda" else 0
N_WORKERS = max(1, min(N_GPU, CFG["N_SEEDS"]))

def train_on_gpu(worker):
    gpu = worker if N_GPU > 1 else None
    return [(seed, *train_model(CFG, seed, gpu)) for seed in range(worker, CFG["N_SEEDS"], N_WORKERS)]

t0 = time.time()
if N_WORKERS > 1:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(N_WORKERS) as pool:
        done = [r for f in [pool.submit(train_on_gpu, w) for w in range(N_WORKERS)] for r in f.result()]
else:
    done = train_on_gpu(0)
done.sort(key=lambda r: r[0])
MODELS = [net for _, net, _, _ in done]
if CKPT_FROM != TARGET:
    ck_species = list(torch.load(ckpt_path(done[0][0]), map_location="cpu", weights_only=False)["cfg"]["SPECIES"])
    assert TARGET in ck_species, f"в весах {CKPT_FROM} нет головы для {TARGET}: {ck_species}"
    HEAD = ck_species.index(TARGET)
    print(f"веса {CKPT_FROM}, головы вида {TARGET}: индекс {HEAD} из {ck_species}")
RUNS = {f"seed {seed}": (h, e) for seed, _, h, e in done}
for sp in SPECIES:   # дальше нужен только целевой вид: освобождаем память под предсказание
    if sp != TARGET: DS[sp].update(genome=None, in_wl=None, track=None)
import gc; gc.collect()
print(f"обучение заняло {(time.time() - t0) / 3600:.2f} ч")

fig, ax = plt.subplots(1, 4, figsize=(18, 3.2))
for name, (h, e) in RUNS.items():
    h, e = pd.DataFrame(h), pd.DataFrame(e)
    if len(h):
        ax[0].plot(h.step, h.loss_prof.rolling(50, min_periods=1).mean(), label=name)
        ax[1].plot(h.step, h.loss_cnt.rolling(50, min_periods=1).mean(), label=name)
        ax[2].plot(h.step, h.lr)
    if len(e): ax[3].plot(e.step, e.AUPRC_ema, "o-", label=f"{name} EMA")
ax[0].set_title("loss профиля"); ax[1].set_title("loss счёта (MSE log1p)"); ax[2].set_title("LR"); ax[3].set_title("AUPRC на tune")
for a in ax: a.set_xlabel("шаг")
ax[0].legend(fontsize=7); ax[3].legend(fontsize=7); plt.tight_layout(); plt.show()
''')

md(r'''
## 7. Предсказание отложенных контигов

Окна идут с шагом `WIN − 2·CROP`, берётся центральная зона; нарезка повторяется `TTA_SHIFTS` раз со сдвигом, log-rate усредняется.
Нормировка профиля — по позициям whitelist окна (как в обучении).
''')
code(r'''
# 7.1 Строки шаблона и предсказание по контигам (как в v2, плюс маска whitelist для нормировки профиля)
def make_rows(df):
    r = df[["chrom", "start", "end"]].copy(); st = df["strand"]
    r["strand"] = st.values.astype(int) if pd.api.types.is_integer_dtype(st) else (st.astype(str).values == "-").astype(int)
    r["len"] = r.end - r.start; r["off"] = np.r_[0, np.cumsum(r.len.values)[:-1]]
    return r.reset_index(drop=True)

def train_rows(contigs):
    iv = T["wl"][T["wl"].chrom.isin(set(contigs))]
    return make_rows(pd.concat([iv.assign(strand=0), iv.assign(strand=1)]))

def gather_rows(rows, contig_fn, dtype=np.float32, desc=None):
    out = np.zeros(int(rows.len.sum()), dtype)
    groups = rows.groupby("chrom", sort=False)
    for c, g in tqdm(groups, total=groups.ngroups, desc=desc, disable=desc is None, mininterval=5):
        lo, hi = int(g.start.min()), int(g.end.max()); arr = contig_fn(c, g, lo, hi)
        for s, e, st, o in zip(g.start.values, g.end.values, g.strand.values, g.off.values):
            out[o:o + e - s] = arr[st, s - lo:e - lo]
    return out

def rows_nmask(rows):
    return gather_rows(rows, lambda c, g, lo, hi: np.broadcast_to(T["genome"][c][lo:hi] == 8, (2, hi - lo)), bool)

def rows_targets(rows):
    def fn(c, g, lo, hi):
        a = np.zeros((2, hi - lo), np.float32); t = T["track"].get(c)
        if t is not None:
            k = (t["pos"] >= lo) & (t["pos"] < hi); a[t["strand"][k], t["pos"][k] - lo] = t["tot"][k]
        return a
    return gather_rows(rows, fn)

@torch.no_grad()
def predict_contig(models, c, g, lo, hi, cfg):
    W, C = cfg["WIN"], cfg["CROP"]; step = W - 2 * C
    iv = np.unique(np.stack([g.start.values, g.end.values], 1), axis=0)
    ms, me = iv[:, 0], np.maximum.accumulate(iv[:, 1])
    def overlaps(a, b):
        k = np.searchsorted(me, a, side="right"); return k < len(ms) and ms[k] < b
    starts = [s for k in range(cfg["TTA_SHIFTS"]) for s in range(lo - C - k * step // cfg["TTA_SHIFTS"], hi - C, step) if overlaps(s + C, s + C + step)]
    acc = np.zeros((2, hi - lo), np.float32); cnt = np.zeros(hi - lo, np.float32)
    for i in range(0, len(starts), cfg["PRED_BATCH"]):
        chunk = starts[i:i + cfg["PRED_BATCH"]]
        lr = predict_windows(models, np.stack([window_codes(T, c, s, W) for s in chunk]), np.stack([window_mask(T, c, s, W) for s in chunk]), cfg)
        for s, p in zip(chunk, lr):
            a, b = max(s + C, lo), min(s + C + step, hi)
            if b > a: acc[:, a - lo:b - lo] += p[:, a - s - C:b - s - C]; cnt[a - lo:b - lo] += 1
    return acc / np.maximum(cnt, 1)

def predict_rows(models, rows, cfg=CFG, desc="предсказание"):
    return gather_rows(rows, lambda c, g, lo, hi: predict_contig(models, c, g, lo, hi, cfg), desc=desc)

def per_model(fn):
    if N_WORKERS > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(MODELS)) as pool: return list(pool.map(lambda im: fn(*im), enumerate(MODELS)))
    return [fn(i, m) for i, m in enumerate(MODELS)]
''')

code(r'''
# 7.2 tune и report: каждая модель и ансамбль v3; подбор alpha на tune
# Размер теста (не-N позиций): alpha подбираем и оцениваем с рангами, растянутыми до этого размера (см. rank_power)
TEST_ROWS = make_rows(T["tpl"]); TEST_N = rows_nmask(TEST_ROWS); TEST_LEN = len(TEST_N)
N_REF = int((~TEST_N).sum())
print(f"тест: {TEST_LEN:,} позиций, не-N {N_REF:,}: под этот размер подбирается alpha")

def choose_alpha(score, gt, nmask, alphas=CFG["ALPHAS"]):
    sel = ~nmask
    grid = pd.DataFrame([{"alpha": a, **official_score(rank_power(score[sel], a, n_ref=N_REF), gt[sel])} for a in alphas]).set_index("alpha")
    return float(grid.index[grid.AUPRC >= CFG["KEEP_FRAC"] * grid.AUPRC.max()].max()), grid

def evaluate(score, gt, nmask, alpha):
    return {**{f"{k} (сырое)": v for k, v in raw_score(score, gt, ~nmask).items()}, **official_score(rank_power(score, alpha, nmask, n_ref=N_REF), gt)}

# EPIC_REUSE_V3=1: взять готовые предсказания v3 (v3_holdout_*_lograte.npy, v3_test_lograte.npy) из Input — GPU не нужен
def find_v3_cache():
    if os.environ.get("EPIC_REUSE_V3") != "1": return None
    need = ("v3_holdout_tune_lograte.npy", "v3_holdout_report_lograte.npy", "v3_test_lograte.npy")
    for p in glob.glob(f"/kaggle/input/**/{TARGET}/v3_test_lograte.npy", recursive=True):   # только папка целевого вида
        d = os.path.dirname(p)
        if all(os.path.exists(f"{d}/{f}") for f in need): return d
    return None
V3_CACHE = find_v3_cache(); print("готовые предсказания v3:", V3_CACHE)

def show(df, name):
    # таблица видна и в HTML, и в логе Kaggle (display в лог не попадает), и сохраняется в CSV
    display(df); print(f"--- {name}\n{df.to_string()}"); df.to_csv(f"{OUT_DIR}/{name}.csv")

def cap_rows(rows, max_bp, piece=1_000_000, seed=0):
    # выборка кусков whitelist (до piece нт, обе цепи) суммарно до max_bp на цепь; детерминирована (seed), одинакова во всех прогонах
    plus = rows[rows.strand == 0]
    starts = [np.arange(s, e, piece) for s, e in zip(plus.start.values, plus.end.values)]
    ch = pd.DataFrame({"chrom": np.repeat(plus.chrom.values, [len(a) for a in starts]), "start": np.concatenate(starts)})
    ch["end"] = np.minimum(ch.start + piece, np.repeat(plus.end.values, [len(a) for a in starts]))
    ch = ch.sample(frac=1, random_state=seed); ch = ch[np.cumsum((ch.end - ch.start).values) <= max_bp].sort_values(["chrom", "start"])
    return make_rows(pd.concat([ch.assign(strand=0), ch.assign(strand=1)]))

HO = {}
for gname in ("tune", "report"):
    rows = train_rows(GROUPS[gname])
    if CFG.get("HO_MAX_BP") and rows.len.sum() / 2 > CFG["HO_MAX_BP"]:
        full = rows.len.sum() / 2; rows = cap_rows(rows, CFG["HO_MAX_BP"])
        print(f"{gname}: оцениваем на выборке {rows.len.sum() / 2 / 1e6:.0f} из {full / 1e6:.0f} Мб (HO_MAX_BP)")
    HO[gname] = dict(rows=rows, gt=rows_targets(rows), n=rows_nmask(rows))
    if V3_CACHE: HO[gname]["preds"] = [np.load(f"{V3_CACHE}/v3_holdout_{gname}_lograte.npy")]
    else: HO[gname]["preds"] = per_model(lambda i, m: predict_rows([m], rows, desc=f"{gname}, модель {i}"))

# Отбор моделей: сид, который разошёлся при обучении, портит среднее. Оставляем модели с AUPRC на tune >= KEEP_MODEL_FRAC * лучшей
tn = HO["tune"]
seed_auprc = [raw_score(p, tn["gt"], ~tn["n"])["AUPRC"] for p in tn["preds"]]
KEEP = [i for i, a in enumerate(seed_auprc) if a >= CFG.get("KEEP_MODEL_FRAC", 0.8) * max(seed_auprc)]
print("AUPRC (сырое) на tune по моделям:", np.round(seed_auprc, 4), "-> в ансамбле модели", KEEP)
MODELS_ALL, MODELS = MODELS, ([MODELS[i] for i in KEEP] if not V3_CACHE else MODELS)
for gname in ("tune", "report"):
    HO[gname]["v3"] = np.mean([HO[gname]["preds"][i] for i in KEEP], axis=0)
    np.save(f"{OUT_DIR}/v3_holdout_{gname}_lograte.npy", HO[gname]["v3"])

ALPHA_V3, grid = choose_alpha(HO["tune"]["v3"], HO["tune"]["gt"], HO["tune"]["n"])
show(grid.round(5), "v3_alpha_grid_tune"); print("alpha v3 =", ALPHA_V3)
rep = HO["report"]
table = {f"v3 seed {i}": evaluate(p, rep["gt"], rep["n"], ALPHA_V3) for i, p in enumerate(rep["preds"])}
table[f"v3 ансамбль {KEEP}"] = evaluate(rep["v3"], rep["gt"], rep["n"], ALPHA_V3)
show(pd.DataFrame(table).T.round(4), "v3_report_metrics")
''')

md(r'''
## 8. Смешивание с v2

Ищем папку с файлами v2 (`holdout_tune_lograte.npy`, `holdout_report_lograte.npy`, `test_lograte.npy`, `holdout_groups.json`):
переменная `EPIC_V2_DIR` или любой Input в `/kaggle/input`. Их сохраняет ноутбук v2 (раздел 7.2 и 8.1), если прогнать его для того же вида.

Два способа смешивания: `z` — среднее стандартизованных log-rate, `rank` — среднее нормированных рангов.
Способ и вес v3 выбираем на **tune** по сумме относительных AUPRC и Spearman, затем честно смотрим на **report**.
''')
code(r'''
# 8.1 Подбор смеси на tune, проверка на report
def find_v2():
    need = ("holdout_tune_lograte.npy", "holdout_report_lograte.npy", "test_lograte.npy")
    cands = [os.environ.get("EPIC_V2_DIR", "")] + [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{TARGET}/holdout_tune_lograte.npy", recursive=True)]
    for d in cands:
        if d and all(os.path.exists(f"{d}/{f}") for f in need): return d
    return None

def normalise(x, nmask, how):
    ok = ~nmask
    if how == "z":
        v = x.astype(np.float64); mu, sd = v[ok].mean(), v[ok].std() + 1e-9
        return (v - mu) / sd
    r = np.empty(len(x), np.float64); order = np.argsort(x, kind="stable")
    r[order] = np.arange(len(x), dtype=np.float64) / len(x)
    return r

def blend(a, b, w, how, nmask):
    return w * normalise(a, nmask, how) + (1 - w) * normalise(b, nmask, how)

# --- правило выбора по лидерборду: место вида = порядок по ПРОИЗВЕДЕНИЮ мест по AUPRC и Spearman среди всех сабмитов
#     (проверено 2026-10-16 по API: порядок команд совпадает с rank0_AUPRC * rank0_Spearman). Метрики кандидата берём с tune
#     (квантование с рангами под размер теста), считаем его места среди чужих сабмитов и берём минимум произведения.
def lb_submissions(species):
    import urllib.request
    try:
        sp = json.load(urllib.request.urlopen("https://epic.autosome.org/api/v1/species", timeout=30))
        ids = [x.get("id") or x.get("assembly") for x in (sp if isinstance(sp, list) else sp.get("species", sp.get("items", [])))]
        sid = next((i for i in ids if i and (i == species or i.startswith(species + "-"))), species)
        d = json.load(urllib.request.urlopen(f"https://epic.autosome.org/api/v1/leaderboard/species/{sid}?limit=500&show=all", timeout=60))
        out = []
        for t in d["teams"]:
            m = {x["metric"]: x["score"] for x in t["metrics"]}; out.append((t["teamName"], m.get("prauc"), m.get("spearman")))
        out = [(t, a, r) for t, a, r in out if a is not None and r is not None and t != CFG["LB_TEAM"]]
        print(f"лидерборд {sid}: {len(out)} чужих сабмитов"); return out
    except Exception as e:
        print("лидерборд недоступен:", repr(e)[:200]); return None

def lb_rank(auprc, spear, subs):
    auprc, spear = auprc * CFG["LB_CALIB"][0], spear * CFG["LB_CALIB"][1]
    A = np.array([a for _, a, _ in subs]); S = np.array([r for _, _, r in subs])
    ra, rs = 1 + int((A > auprc).sum()), 1 + int((S > spear).sum())
    return ra, rs, ra * rs

def alpha_grid_ranked(score, gt, nmask, alphas):
    # сетка alpha с одной сортировкой: ранги считаются один раз, затем r^-alpha для каждого alpha (как rank_power с n_ref)
    sel = ~nmask; x = np.asarray(score, np.float64)[sel]; n = len(x)
    order = np.argsort(-x, kind="stable"); r = np.empty(n, np.float64); r[order] = np.arange(1, n + 1, dtype=np.float64); del order
    g = gt[sel]; rows_ = []
    for a in alphas:
        v = (1 - (r - 1) / n) if a == 0 else np.power(r * (N_REF / n), -a)
        rows_.append({"alpha": a, **official_score(np.rint(np.clip(v, 0, 1) * SCALE) / SCALE, g)})
    return pd.DataFrame(rows_)

LB_SUBS = lb_submissions(TARGET) if CFG.get("SELECT") == "lb" else None

V2_DIR = find_v2(); BLEND = None
# EPIC_BLEND="z:0.4" (способ:вес v3) и EPIC_ALPHA="0.7" — задать смесь вручную, когда tune нечестен для v2 (v2 училась на всех контигах вида)
FORCE_BLEND = os.environ.get("EPIC_BLEND", "")
FORCE_ALPHA = float(os.environ["EPIC_ALPHA"]) if os.environ.get("EPIC_ALPHA") else None
if V2_DIR:
    g2 = f"{V2_DIR}/holdout_groups.json"
    same_groups = (not os.path.exists(g2)) or json.load(open(g2)) == GROUPS
    V2 = {g: np.load(f"{V2_DIR}/holdout_{g}_lograte.npy") for g in ("tune", "report")}
    if same_groups and all(len(V2[g]) == len(HO[g]["gt"]) for g in V2):
        tn = HO["tune"]; sel = ~tn["n"]; rows_ = []
        for how in ("z", "rank"):
            for w in CFG["BLEND_WEIGHTS"]:
                s = blend(HO["tune"]["v3"], V2["tune"], w, how, tn["n"])
                rows_.append({"how": how, "w_v3": w, **raw_score(s[sel], tn["gt"][sel])})
        bg = pd.DataFrame(rows_)
        bg["score"] = bg.AUPRC / bg.AUPRC.max() + bg.Spearman / bg.Spearman.max()
        best = bg.loc[bg.score.idxmax()]; BLEND = dict(how=best.how, w=float(best.w_v3))
        show(bg.round(5), "blend_grid_tune")
        LB_PICK = None
        if LB_SUBS and not FORCE_BLEND:
            # кандидаты: лучшие по сырой сетке смеси + v2 и v3 по отдельности; для каждого сетка alpha; минимум произведения мест
            short = bg.nlargest(CFG["LB_SHORTLIST"], "score")[["how", "w_v3"]].values.tolist() + [["z", 0.0], ["z", 1.0]]
            lr = []
            for how, w in {(h, float(w)) for h, w in short}:
                g = alpha_grid_ranked(blend(HO["tune"]["v3"], V2["tune"], w, how, tn["n"]), tn["gt"], tn["n"], CFG["ALPHAS"])
                for _, row in g.iterrows():
                    ra, rs, pr = lb_rank(row.AUPRC, row.Spearman, LB_SUBS)
                    lr.append({"how": how, "w_v3": w, "alpha": row.alpha, "AUPRC": row.AUPRC, "Spearman": row.Spearman, "место AUPRC": ra, "место Spearman": rs, "произведение": pr})
            lr = pd.DataFrame(lr).sort_values(["произведение", "Spearman"], ascending=[True, False])
            show(lr.head(15).round(5), "lb_select_tune")
            LB_PICK = lr.iloc[0]; BLEND = dict(how=LB_PICK.how, w=float(LB_PICK.w_v3))
            print(f"ИТОГ выбор по лидерборду: {BLEND}, alpha {LB_PICK.alpha}, tune {LB_PICK.AUPRC:.4f}/{LB_PICK.Spearman:.4f}, "
                  f"места {LB_PICK['место AUPRC']}x{LB_PICK['место Spearman']}")
        if FORCE_BLEND:   # v2 видела tune этого вида: сетка на tune смещена в пользу v2, вес и alpha задаём заранее
            BLEND = dict(how=FORCE_BLEND.split(":")[0], w=float(FORCE_BLEND.split(":")[1]))
            print("смесь задана вручную (EPIC_BLEND):", BLEND, "| по tune была бы:", dict(how=best.how, w=float(best.w_v3)))
        else:
            print("выбрана смесь:", BLEND)
        tune_bl = blend(HO["tune"]["v3"], V2["tune"], BLEND["w"], BLEND["how"], tn["n"])
        ALPHA, grid_bl = choose_alpha(tune_bl, tn["gt"], tn["n"])
        if LB_PICK is not None: ALPHA = float(LB_PICK.alpha)
        if FORCE_ALPHA is not None: ALPHA = FORCE_ALPHA
        show(grid_bl.round(5), "blend_alpha_grid_tune")
        ALPHA_V2, _ = choose_alpha(V2["tune"], tn["gt"], tn["n"])
        rep_bl = blend(rep["v3"], V2["report"], BLEND["w"], BLEND["how"], rep["n"])
        cmp_ = pd.DataFrame({"v2": evaluate(V2["report"], rep["gt"], rep["n"], ALPHA_V2),
                             "v3": evaluate(rep["v3"], rep["gt"], rep["n"], ALPHA_V3),
                             f"смесь {BLEND['how']} w_v3={BLEND['w']}": evaluate(rep_bl, rep["gt"], rep["n"], ALPHA),
                             **{f"смесь, alpha={a}": evaluate(rep_bl, rep["gt"], rep["n"], a) for a in (0.7, 0.8, 0.9)}}).T
        show(cmp_.round(4), "report_comparison"); print("alpha смеси =", ALPHA)
    else:
        print("v2 найден, но группы/длины не совпадают с v3 -> смешивание пропущено")
if BLEND is None:
    ALPHA = ALPHA_V3
    print("сабмит будет только из v3" + ("" if V2_DIR else " (предсказания v2 не найдены)"))
''')

md("## 9. Тест и сабмит")
code(r'''
# 9.1 Предсказание теста, смесь, офлайн-оценка (если есть ответы, т.е. Nematostella), файл сабмита
TEST_V3 = np.load(f"{V3_CACHE}/v3_test_lograte.npy") if V3_CACHE else predict_rows(MODELS, TEST_ROWS, desc="тест")
assert len(TEST_V3) == TEST_LEN
np.save(f"{OUT_DIR}/v3_test_lograte.npy", TEST_V3)
FINAL = TEST_V3
if BLEND is not None:
    v2t = np.load(f"{V2_DIR}/test_lograte.npy"); assert len(v2t) == TEST_LEN, "длина test_lograte v2 не совпала с шаблоном"
    FINAL = blend(TEST_V3, v2t, BLEND["w"], BLEND["how"], TEST_N); del v2t
np.save(f"{OUT_DIR}/test_nmask.npy", TEST_N)
np.save(f"{OUT_DIR}/test_final_score.npy", FINAL.astype(np.float32))   # смесь до квантования: alpha можно менять без GPU
SUB = rank_power(FINAL, ALPHA, TEST_N)
print(f"тест: {TEST_LEN:,} позиций, N: {int(TEST_N.sum()):,}")

gt_path = f"{BASES[TARGET]}/ground_truth/csRNA-test.txt.gz"
if os.path.exists(gt_path):
    read_col = lambda p, dt: pd.read_csv(p, header=None, dtype=dt)[0].values
    gt = read_col(gt_path, np.float32); assert len(gt) == TEST_LEN
    m1 = read_col(f"{BASES[TARGET]}/scoring/mask1-test.txt.gz", np.int8).astype(bool)
    m2 = read_col(f"{BASES[TARGET]}/scoring/mask2-test.txt.gz", np.int8).astype(bool)
    show(pd.DataFrame({nm: official_score(SUB, gt, mk) for nm, mk in (("half1", m1), ("half2", m2), ("весь тест", None))}).T.round(4), "test_offline")

sub_path = f"{OUT_DIR}/{TARGET}.predictions.txt.gz"
ints = np.rint(SUB.astype(np.float64) * SCALE).astype(np.int32)

def write_submission(ints, path, chunk=8_000_000):
    with gzip.open(path, "wb", compresslevel=4) as fh:
        for a in tqdm(range(0, len(ints), chunk), desc="пишу сабмит"):
            v = ints[a:a + chunk]; buf = np.empty((len(v), 8), np.uint8)
            buf[:, 0] = ord("0") + v // SCALE; buf[:, 1] = ord("."); buf[:, 7] = ord("\n")
            frac = v % SCALE
            for k in range(6, 1, -1): buf[:, k] = ord("0") + frac % 10; frac //= 10
            fh.write(buf.tobytes())

write_submission(ints, sub_path)
for a in CFG.get("EXTRA_SUB_ALPHAS", ()):
    if a == ALPHA: continue
    extra = f"{OUT_DIR}/{TARGET}.predictions.alpha{a:.2f}.txt.gz"
    write_submission(np.rint(rank_power(FINAL, a, TEST_N).astype(np.float64) * SCALE).astype(np.int32), extra)
    print("дополнительный сабмит:", extra)
del FINAL
with gzip.open(sub_path, "rt") as fh: head = [next(fh).strip() for _ in range(3)]
assert head == [f"{v / SCALE:.5f}" for v in ints[:3]]
checks = {"строк = позиций шаблона": len(SUB) == TEST_LEN, "значения в [0, 1]": bool((SUB >= 0).all() and (SUB <= 1).all()),
          "N-позиции = 0": bool((SUB[TEST_N] == 0).all()), "различных уровней": int(len(np.unique(ints))), "alpha": ALPHA,
          "смесь": str(BLEND), "размер файла, МБ": round(os.path.getsize(sub_path) / 1e6, 1)}
show(pd.Series(checks, name="проверка").to_frame(), "submission_checks")
assert checks["строк = позиций шаблона"] and checks["значения в [0, 1]"]
print("сабмит:", sub_path)
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code": cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(root, "epic_v3_bpnet.ipynb"))
    print("written epic_v3_bpnet.ipynb,", len(CELLS), "cells")
