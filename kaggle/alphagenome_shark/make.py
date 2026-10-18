# Собирает kaggle/alphagenome_shark/run.py: проверка AlphaGenome на отложенных контигах report акулы sSquSuc5.hap2 против v2 ft2.
# Вход (датасет аккаунта ivanmitochondrion): holdout_report_lograte.npy + holdout_groups.json из epic-v2-ft2 (акула). Секрет ALPHAGENOME_API_KEY.
# Запуск: python kaggle/alphagenome_shark/make.py  (затем /tmp/kpush.py kaggle/alphagenome_shark)
import json, os
root = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(os.path.dirname(root))
nb = json.load(open(f"{repo}/epic_v2_model.ipynb", encoding="utf-8"))
metrics = "".join(nb["cells"][19]["source"])                       # официальные метрики + rank_power, точная копия
ag = open(f"{repo}/tools/alphagenome_validate.py", encoding="utf-8").read()
ag = ag[ag.index('import os\nimport numpy as np'):]
driver = open(f"{root}/driver.py", encoding="utf-8").read()
head = "import os, gc, glob, gzip, json, tarfile, requests\nimport numpy as np, pandas as pd\nfrom sklearn.metrics import average_precision_score\n"
open(f"{root}/run.py", "w", encoding="utf-8").write(head + "\n" + metrics + "\n" + ag + "\n" + driver)
json.dump({"id": "ivanmitochondrion/epic-alphagenome-shark", "title": "epic-alphagenome-shark", "code_file": "run.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": ["ivanmitochondrion/epic-shark-ft2-report"], "competition_sources": [], "kernel_sources": []},
          open(f"{root}/kernel-metadata.json", "w"), indent=1)
print("готово")
