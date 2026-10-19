# Собирает ядро Kaggle из epic_v3_bpnet.ipynb (мультивидовой BPNet v3): первая ячейка после заголовка задаёт переменные окружения.
# python kaggle/v3_ms/make.py <slug> '<json env>' [datasets] [--cpu] [--prep-only] [--train-only] [--kernels=a,b] [--account=ivanmitochondrion]
# --prep-only: только разбор данных видов + кэш (EPIC_PREP_CACHE=1), бесплатное CPU-ядро; --train-only: до ячейки 6.4 включительно (без предсказаний)
import json, os, sys
slug, envj = sys.argv[1], sys.argv[2]
datasets = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] and not sys.argv[3].startswith("--") else []
gpu = "--cpu" not in sys.argv; prep_only = "--prep-only" in sys.argv; train_only = "--train-only" in sys.argv
kernels = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--kernels=")), [])
account = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--account=")), "ivanmitochondrion")
root = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(os.path.dirname(root))
nb = json.load(open(f"{repo}/epic_v3_bpnet.ipynb")); env_d = json.loads(envj)
if prep_only: env_d["EPIC_PREP_CACHE"] = "1"
if train_only: env_d["EPIC_TRAIN_ONLY"] = "1"
env = "import os\n" + "".join(f"os.environ[{k!r}] = {str(v)!r}\n" for k, v in env_d.items()) + 'print("env:", {k: v for k, v in os.environ.items() if k.startswith("EPIC_")})'
first = lambda pre: next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith(pre))
if prep_only: nb["cells"] = nb["cells"][:first("# 2.1 ") + 1]
if train_only: nb["cells"] = nb["cells"][:first("# 6.4 ") + 1]
nb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": env.splitlines(keepends=True)})
out = f"{root}/{slug}"; os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/notebook.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump({"id": f"{account}/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True, **({"machine_shape": "NvidiaTeslaT4"} if gpu else {}),
           "dataset_sources": datasets, "competition_sources": [], "kernel_sources": kernels}, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("готово:", out)
