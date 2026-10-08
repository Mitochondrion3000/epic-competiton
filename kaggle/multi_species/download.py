# скачать и распаковать архив вида с Zenodo (используется из run.py для Ofas_2.0)
import hashlib, tarfile, requests, os
def _dl(species, data_dir):
    rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == f"{species}.tgz"][0]
    dst, md5 = f"{data_dir}/{species}.tgz", hashlib.md5()
    os.makedirs(data_dir, exist_ok=True)
    with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(dst, "wb") as fh:
        for chunk in r.iter_content(1 << 22): fh.write(chunk); md5.update(chunk)
    assert md5.hexdigest() == f["checksum"].split(":")[-1]
    with tarfile.open(dst) as tf: tf.extractall(data_dir, filter="data")
    os.remove(dst)
_dl("Ofas_2.0", DATA)
