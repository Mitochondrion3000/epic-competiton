# Kaggle-скрипт: шаг 7 (шаг 6 + непрерывные счета мотивов и признаки режима) для jaNemVect1.1 на CPU.
# Код — из приватного датасета epic-simple-code (репозиторий приватный), входы — датасет epic-simple-nem-inputs.
import glob, os, shutil, subprocess, sys, time
SPECIES = os.environ.get("EPIC_SPECIES", "jaNemVect1.1")
for src in glob.glob(f"/kaggle/input/**/{SPECIES}.*", recursive=True):
    if not src.endswith((".run.ipynb", "_features.parquet")):
        shutil.copy(src, "/kaggle/working/"); print("взят вход:", os.path.basename(src))
code = glob.glob("/kaggle/input/**/step7_final.ipynb", recursive=True)
assert code, "в датасете epic-simple-code нет step7_final.ipynb"
os.makedirs("/tmp/repo/simple_model", exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(code[0]), "*")):
    if f.endswith((".py", ".ipynb")): shutil.copy(f, "/tmp/repo/simple_model/")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pyarrow"], check=False)
env = dict(os.environ, EPIC_SPECIES=SPECIES, EPIC_DATA_DIR="/tmp/epic", EPIC_OUT_DIR="/kaggle/working")
t0 = time.time(); step = "step7_final"
r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1",
                    f"/tmp/repo/simple_model/{step}.ipynb", "--output", f"/kaggle/working/{SPECIES}.{step}.run.ipynb"],
                   cwd="/tmp/repo/simple_model", env=env, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
if r.returncode != 0: sys.exit(f"{step} упал (код {r.returncode})")
print(f"=== {step}: готово ({(time.time() - t0) / 60:.0f} мин)")
