# Kaggle-скрипт: шаги 1-3 простой модели для Ofas_2.0 на CPU.
# Код берётся из репозитория (ветка simple_model), данные качаются с Zenodo, результаты пишутся в /kaggle/working.
import os, subprocess, sys, time
BRANCH = "claude/gracious-brown-4ybsmj"
REPO = "https://github.com/Mitochondrion3000/epic-competiton.git"
SPECIES = os.environ.get("EPIC_SPECIES", "Ofas_2.0")
STEPS = ["step1_data_table", "step2_features", "step3_motif_db_and_model"]

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
