import glob, json, os, shutil, subprocess, sys, time
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
# код модели: из приватного датасета Kaggle epic-simple-code (репозиторий может быть приватным); иначе — из GitHub
code = glob.glob("/kaggle/input/**/final.py", recursive=True)
if code:
    os.makedirs("/tmp/repo/simple_model", exist_ok=True)
    for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*.py")): shutil.copy(f, "/tmp/repo/simple_model/")
else:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, "https://github.com/Mitochondrion3000/epic-competiton.git", "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4", "pyarrow"], check=False)
sys.path.insert(0, "/tmp/repo/simple_model")
import lightgbm as lgb
import features as F, scoring as S, final as FN
FP = f"{DATA}/{SPECIES}/for_participants"
FN.download_dataset(SPECIES, DATA)
params, BASE, ALL = FN.load_params(f"{OUT}/{SPECIES}.step3b_params.json")
N_PARTS = 3

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
wl = pd.read_csv(f"{FP}/whitelist.train.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], dtype={"chrom": str})
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
