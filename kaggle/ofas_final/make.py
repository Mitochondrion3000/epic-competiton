# Генерирует скрипты и метаданные Kaggle для финальной модели Ofas_2.0: train -> predict (3 части параллельно) -> merge.
# Запуск: python kaggle/ofas_final/make.py  (создаёт папки kaggle/ofas_final/{train,p0,p1,p2,merge})
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
USER, N_PARTS = "ivanmitochondrion", 3

BOOT = '''import glob, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
BRANCH = "claude/gracious-brown-4ybsmj"
SPECIES = "Ofas_2.0"
OUT, DATA = "/kaggle/working", "/tmp/epic"
T0 = time.time()
def log(*a): print(f"[{(time.time() - T0) / 60:6.1f} мин]", *a, flush=True)
# входы предыдущих ноутбуков -> рабочая папка
for src in glob.glob(f"/kaggle/input/**/{SPECIES}.*", recursive=True):
    if not src.endswith((".run.ipynb", "_features.parquet", ".npy")) and not os.path.exists(f"{OUT}/{os.path.basename(src)}"):
        shutil.copy(src, OUT)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, "https://github.com/Mitochondrion3000/epic-competiton.git", "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4", "pyarrow"], check=False)
sys.path.insert(0, "/tmp/repo/simple_model")
import lightgbm as lgb
import features as F, scoring as S, final as FN
FP = f"{DATA}/{SPECIES}/for_participants"
FN.download_dataset(SPECIES, DATA)
params, BASE, ALL = FN.load_params(f"{OUT}/{SPECIES}.step3b_params.json")
N_PARTS = %d
''' % N_PARTS

TRAIN = BOOT + '''
# ---------- шаг 1 из 3: обучение ----------
MAX_ROWS = 7_000_000
table = pd.read_parquet(f"{OUT}/{SPECIES}.step1_table.parquet"); meta = json.load(open(f"{OUT}/{SPECIES}.step1_meta.json"))
rows = table if len(table) <= MAX_ROWS else pd.concat([table[table.y == 1], table[table.y == 0].sample(MAX_ROWS - int(table.y.sum()), random_state=0)]).sort_index()
rows = rows.reset_index(drop=True)
SEQ, LOW = FN.read_genome(f"{DATA}/{SPECIES}/genome/genome.fa.gz", set(rows.chrom) | set(meta["VAL_CONTIGS"]))
log("строк", len(rows), "контигов", len(SEQ))
XB, XE = FN.build_rows(rows, SEQ, LOW, params, BASE, len(ALL) - len(BASE), log=log)
y, tot = rows.y.values, rows.tot.values
log("признаки готовы", XB.shape, XE.shape)

# честная оценка: модель на fit -> плотная валидация -> alpha
m_fit = [FN.train_pair(XB, XE, y, tot, np.flatnonzero(rows.split.values == "fit"), ALL, 0)]
log("модель на fit обучена")
wl = pd.read_csv(f"{FP}/whitelist.train.bed.gz", sep="\\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], dtype={"chrom": str})
lab = table[table.y == 1]; DS, DY, n = [], [], 0
for c in meta["VAL_CONTIGS"]:
    L = len(SEQ[c]); inwl = np.zeros(L, bool)
    for s, e in zip(*wl[wl.chrom == c][["start", "end"]].values.T): inwl[s:e] = True
    inwl[L // 2:] = False; dpos = np.flatnonzero(inwl)
    if not len(dpos): continue
    lc = lab[lab.chrom == c]; t = dict(zip(lc.pos.values * 2 + (lc.strand.values == "-"), lc.tot.values))
    for st in "+-":
        DS.append(FN.predict_positions(m_fit, SEQ[c], LOW[c], dpos, st, params, BASE))
        DY.append(np.array([t.get(k, 0.0) for k in (dpos * 2 + (st == "-")).tolist()], np.float32))
    n += 2 * len(dpos)
    if n >= 12_000_000: break
DS, DY = np.concatenate(DS), np.concatenate(DY)
grid = pd.DataFrame([{"alpha": a, **S.official_score(S.rank_power(DS, a), DY)} for a in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7)]).set_index("alpha")
grid["rel"] = grid.AUPRC / grid.AUPRC.max()
ALPHA = float(grid[grid.rel >= 0.995].Spearman.idxmax())
log("плотная валидация", len(DY), "стартов", int((DY > 0).sum())); print(grid.to_string(), flush=True)
log(f"alpha = {ALPHA}: AUPRC {grid.loc[ALPHA, 'AUPRC']:.4f}, Spearman {grid.loc[ALPHA, 'Spearman']:.4f}")
del m_fit, DS

# финальные модели на всех строках
for seed in range(3):
    c, r = FN.train_pair(XB, XE, y, tot, np.arange(len(rows)), ALL, seed)
    c.save_model(f"{OUT}/{SPECIES}.step6_cls_seed{seed}.txt"); r.save_model(f"{OUT}/{SPECIES}.step6_reg_seed{seed}.txt")
    log("модель", seed, "сохранена")
json.dump({"alpha": ALPHA, "train_rows": len(rows), "dense": {"AUPRC": grid.loc[ALPHA, "AUPRC"], "Spearman": grid.loc[ALPHA, "Spearman"]},
           "grid": grid.reset_index().to_dict("records")}, open(f"{OUT}/{SPECIES}.step6_summary.json", "w"), default=float)
log("готово")
'''

PREDICT = BOOT + '''
# ---------- шаг 2 из 3: предсказание части %d теста ----------
PART = %d
models = [(lgb.Booster(model_file=f"{OUT}/{SPECIES}.step6_cls_seed{s}.txt"), lgb.Booster(model_file=f"{OUT}/{SPECIES}.step6_reg_seed{s}.txt")) for s in range(3)]
tpl = FN.read_template(f"{FP}/template.test.bed.gz")
part_of = FN.test_groups(tpl, N_PARTS)
mine = [c for c, k in part_of.items() if k == PART]
SEQ, LOW = FN.read_genome(f"{DATA}/{SPECIES}/genome/genome.fa.gz", set(mine))
log("часть", PART, "контигов", len(mine), "позиций", int(tpl[tpl.chrom.isin(mine)].len.sum()))
out = FN.predict_part(models, tpl, part_of, PART, SEQ, LOW, params, BASE, log=log)
np.save(f"{OUT}/{SPECIES}.step6_part{PART}.npy", out)
log("готово", len(out))
'''

MERGE = BOOT + '''
# ---------- шаг 3 из 3: сборка сабмита и проверка ----------
summary = json.load(open(f"{OUT}/{SPECIES}.step6_summary.json")); ALPHA = summary["alpha"]
tpl = FN.read_template(f"{FP}/template.test.bed.gz"); N = int(tpl.len.sum())
part_of = FN.test_groups(tpl, N_PARTS)
parts = {k: np.load(glob.glob(f"/kaggle/input/**/{SPECIES}.step6_part{k}.npy", recursive=True)[0]) for k in range(N_PARTS)}
score = FN.assemble(tpl, part_of, parts); del parts
nmask = ~np.isfinite(score)
sub = S.rank_power(score, ALPHA, nmask)
path = f"{OUT}/{SPECIES}.simple_model_final.predictions.txt.gz"
S.write_submission(sub, path)
subprocess.run(["git", "clone", "-q", "https://github.com/autosome-ru/EPIC_scoring.git", "/tmp/EPIC_scoring"], check=True)
v = subprocess.run(["bash", "/tmp/EPIC_scoring/validate", str(N), path], capture_output=True, text=True).stdout.strip()
log(f"сабмит {path}: {os.path.getsize(path) / 1e6:.0f} МБ, строк {N:,}, N-позиций {int(nmask.sum()):,}, alpha {ALPHA}, официальный validate: {v}")
assert v == "ok"
'''

def kernel(name, code, sources):
    d = f"{HERE}/{name}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/run.py", "w").write(code)
    json.dump({"id": f"{USER}/epic-simple-ofas-final-{name}", "title": f"epic-simple-ofas-final-{name}", "code_file": "run.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
               "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": sources},
              open(f"{d}/kernel-metadata.json", "w"), indent=2)

kernel("train", TRAIN, [f"{USER}/epic-simple-ofas-a", f"{USER}/epic-simple-ofas-b"])
for k in range(N_PARTS):
    kernel(f"p{k}", PREDICT % (k, k), [f"{USER}/epic-simple-ofas-b", f"{USER}/epic-simple-ofas-final-train"])
kernel("merge", MERGE, [f"{USER}/epic-simple-ofas-b", f"{USER}/epic-simple-ofas-final-train"] + [f"{USER}/epic-simple-ofas-final-p{k}" for k in range(N_PARTS)])
print("written:", sorted(os.listdir(HERE)))
