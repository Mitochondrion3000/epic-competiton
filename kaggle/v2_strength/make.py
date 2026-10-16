# Собирает ядро Kaggle из epic_v2_model.ipynb: первая ячейка задаёт вид и переопределения CFG через окружение.
# python kaggle/v2_strength/make.py <slug> <species> '<json CFG>'  -> kaggle/v2_strength/<slug>/{notebook.ipynb, kernel-metadata.json}
import json, os, sys
slug, species, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
datasets = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] and not sys.argv[4].startswith("--") else []
holdout_only = "--holdout-only" in sys.argv     # оставить ячейки до 7.2 включительно: без теста и абляции (дешёвая проверка, CPU)
gpu = "--cpu" not in sys.argv
kernels = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--kernels=")), [])
root = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(os.path.dirname(root))
nb = json.load(open(f"{repo}/epic_v2_model.ipynb"))
json.loads(cfg)                                                  # проверка, что JSON корректный
env = (f'import os\nos.environ["EPIC_SPECIES"] = {species!r}\nos.environ["EPIC_CFG_JSON"] = {cfg!r}\n'
       'print("EPIC_CFG_JSON =", os.environ["EPIC_CFG_JSON"])')
if holdout_only:
    k = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("# 7.2 "))
    nb["cells"] = nb["cells"][:k + 1]
nb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": env.splitlines(keepends=True)})
out = f"{root}/{slug}"; os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/notebook.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump({"id": f"ivanmitochondrion/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
           **({"machine_shape": "NvidiaTeslaT4"} if gpu else {}), "dataset_sources": datasets, "competition_sources": [], "kernel_sources": kernels},
          open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("готово:", out)
