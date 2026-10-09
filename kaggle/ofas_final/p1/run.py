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

# ---------- шаг 2 из 3: предсказание части 1 теста ----------
PART = 1
models = [(lgb.Booster(model_file=f"{OUT}/{SPECIES}.step6_cls_seed{s}.txt"), lgb.Booster(model_file=f"{OUT}/{SPECIES}.step6_reg_seed{s}.txt")) for s in range(3)]
tpl = FN.read_template(f"{FP}/template.test.bed.gz")
part_of = FN.test_groups(tpl, N_PARTS)
mine = [c for c, k in part_of.items() if k == PART]
SEQ, LOW = FN.read_genome(f"{DATA}/{SPECIES}/genome/genome.fa.gz", set(mine))
log("часть", PART, "контигов", len(mine), "позиций", int(tpl[tpl.chrom.isin(mine)].len.sum()))
out = FN.predict_part(models, tpl, part_of, PART, SEQ, LOW, params, BASE, log=log)
np.save(f"{OUT}/{SPECIES}.step6_part{PART}.npy", out)
log("готово", len(out))
