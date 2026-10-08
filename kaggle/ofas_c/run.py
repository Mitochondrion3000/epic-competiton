# Kaggle-скрипт: шаг 5 (логотип, тест, сабмит) простой модели для Ofas_2.0 на CPU. Входы — результаты ноутбука epic-simple-ofas-a и epic-simple-ofas-b.
import glob, os, shutil, subprocess, sys, time
BRANCH = "claude/gracious-brown-4ybsmj"
REPO = "https://github.com/Mitochondrion3000/epic-competiton.git"
SPECIES = os.environ.get("EPIC_SPECIES", "Ofas_2.0")
STEPS = ["step5_logo"]

for src in glob.glob(f"/kaggle/input/**/{SPECIES}.*", recursive=True):       # результаты предыдущего ноутбука -> рабочая папка
    if not src.endswith((".run.ipynb", "_features.parquet")):
        shutil.copy(src, "/kaggle/working/"); print("взят вход:", os.path.basename(src))
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pyarrow"], check=False)
env = dict(os.environ, EPIC_SPECIES=SPECIES, EPIC_DATA_DIR="/tmp/epic", EPIC_OUT_DIR="/kaggle/working")
t0 = time.time()
for step in STEPS:
    print(f"=== {step}: старт ({(time.time() - t0) / 60:.0f} мин)", flush=True)
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1",
                        f"/tmp/repo/simple_model/{step}.ipynb", "--output", f"/kaggle/working/{SPECIES}.{step}.run.ipynb"],
                       cwd="/tmp/repo/simple_model", env=env, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
    if r.returncode != 0:
        sys.exit(f"{step} упал (код {r.returncode})")
    print(f"=== {step}: готово ({(time.time() - t0) / 60:.0f} мин)", flush=True)
