# Собирает ядро Kaggle из epic_v2_model.ipynb: первая ячейка задаёт вид и переопределения CFG через окружение.
# python kaggle/v2_strength/make.py <slug> <species> '<json CFG>'  -> kaggle/v2_strength/<slug>/{notebook.ipynb, kernel-metadata.json}
import json, os, sys
slug, species, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
datasets = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []
root = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(os.path.dirname(root))
nb = json.load(open(f"{repo}/epic_v2_model.ipynb"))
json.loads(cfg)                                                  # проверка, что JSON корректный
env = (f'import os\nos.environ["EPIC_SPECIES"] = {species!r}\nos.environ["EPIC_CFG_JSON"] = {cfg!r}\n'
       'print("EPIC_CFG_JSON =", os.environ["EPIC_CFG_JSON"])')
nb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": env.splitlines(keepends=True)})
out = f"{root}/{slug}"; os.makedirs(out, exist_ok=True)
json.dump(nb, open(f"{out}/notebook.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump({"id": f"ivanmitochondrion/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": datasets, "competition_sources": [], "kernel_sources": []},
          open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("готово:", out)
