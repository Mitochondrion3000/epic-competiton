# Собирает ядра I-2: python kaggle/nem_distill/make.py <slug> <K> <fold> [--smoke] [--gpu]
import json, os, sys
slug, K, fold = sys.argv[1], sys.argv[2], sys.argv[3]; smoke = "--smoke" in sys.argv; extra = [a for a in sys.argv if a.startswith("--env=")]; gpu = "--gpu" in sys.argv
root = os.path.dirname(os.path.abspath(__file__)); out = f"{root}/out_{slug}"; os.makedirs(out, exist_ok=True)
src = open(f"{root}/run.py", encoding="utf-8").read()
env = f'import os\nos.environ["EPIC_K"] = "{K}"\nos.environ["EPIC_FOLD"] = "{fold}"\n' + ('os.environ["SMOKE"] = "1"\n' if smoke else "") + "".join('os.environ[%r] = %r\n' % tuple(a[6:].split('=', 1)) for a in extra)
open(f"{out}/run.py", "w", encoding="utf-8").write(env + src)
json.dump({"id": f"ivanmitochondrion/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
           "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True, "dataset_sources": ["ivanmitochondrion/epic-simple-code"],
           "competition_sources": [], "kernel_sources": ["ivanmitochondrion/epic-v2-nematostella"]}, open(f"{out}/kernel-metadata.json", "w"), indent=1)
print("готово:", out)
