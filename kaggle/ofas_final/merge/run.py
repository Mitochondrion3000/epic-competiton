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
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, "https://github.com/Mitochondrion3000/epic-competiton.git", "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4", "pyarrow"], check=False)
sys.path.insert(0, "/tmp/repo/simple_model")
import lightgbm as lgb
import features as F, scoring as S, final as FN
FP = f"{DATA}/{SPECIES}/for_participants"
FN.download_dataset(SPECIES, DATA)
params, BASE, ALL = FN.load_params(f"{OUT}/{SPECIES}.step3b_params.json")
N_PARTS = 3

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
