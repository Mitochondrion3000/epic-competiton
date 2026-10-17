# Собирает kaggle/alphagenome_nem/run.py: метрики EPIC + tools/alphagenome_validate.py + драйвер на Nematostella. Запуск: python kaggle/alphagenome_nem/make.py
# Ядро (CPU, интернет) требует секрет Kaggle ALPHAGENOME_API_KEY (Add-ons -> Secrets), ключ: https://alphagenome.google/api. Бюджет окон: BUDGET (по умолчанию 40 = ~40 Мнт).
import json, os
root = os.path.dirname(os.path.abspath(__file__)); repo = os.path.dirname(os.path.dirname(root))
sc = open(f"{repo}/kaggle/strand_check/run.py", encoding="utf-8").read(); head = sc[:sc.index("import os, glob, json")]
ag = open(f"{repo}/tools/alphagenome_validate.py", encoding="utf-8").read()
ag = ag[ag.index('import os\nimport numpy as np'):]                # без докстринга (он уже описан выше)
rank_power_pad = "\n"
driver = '''
import glob, gzip, tarfile, requests, pandas as pd
BUDGET = int(os.environ.get("AG_BUDGET", "40")); SP = "jaNemVect1.1"; D = "/tmp/epic"; B = f"{D}/{SP}"; OUT = "/kaggle/working"
os.system("pip install -q alphagenome")
client = make_client(); print("ключ найден, клиент создан", flush=True)   # раньше загрузки данных: без секрета падаем сразу
if not os.path.isdir(B):
    os.makedirs(D, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == f"{SP}.tgz"][0]; dst = f"{D}/{SP}.tgz"
    with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(dst, "wb") as fh:
        for ch in r.iter_content(1 << 22): fh.write(ch)
    with tarfile.open(dst) as tf: tf.extractall(D)
    os.remove(dst)
rd = lambda p, dt: pd.read_csv(p, header=None, dtype=dt)[0].values
gt = rd(f"{B}/ground_truth/csRNA-test.txt.gz", np.float32)
m1 = rd(f"{B}/scoring/mask1-test.txt.gz", np.int8).astype(bool); m2 = rd(f"{B}/scoring/mask2-test.txt.gz", np.int8).astype(bool)
tpl = pd.read_csv(f"{B}/for_participants/template.test.bed.gz", sep="\\t", header=None, usecols=range(6), names=["chrom","start","end","name","score","strand"], dtype={"chrom": str, "strand": str})
ln = (tpl.end - tpl.start).values; off = np.r_[0, np.cumsum(ln)[:-1]]; assert ln.sum() == len(gt)
ok = m1 | m2
rows_ok = np.array([ok[o:o + l].any() for o, l in zip(off, ln)])
LUT = np.full(256, 8, np.uint8)
for i, b in enumerate("ACGT"): LUT[ord(b)] = i; LUT[ord(b.lower())] = i + 4
genome, name, ch = {}, None, []; keep = set(tpl.chrom)
def flush():
    if name in keep: genome[name] = LUT[np.frombuffer(b"".join(ch), np.uint8)]
with gzip.open(f"{B}/genome/genome.fa.gz", "rb") as fh:
    for l in fh:
        if l.startswith(b">"): flush(); name = l[1:].split()[0].decode(); ch = []
        elif name in keep: ch.append(l.rstrip())
flush()
wins = select_windows(tpl, rows_ok, BUDGET); print("окон:", len(wins), "~Мнт:", len(wins) * W / 1e6, flush=True)
cache = {}
for i in range(0, len(wins), 5):                             # по 5 окон, чтобы не терять прогресс при сбое
    predict_windows(client, genome, wins[i:i + 5], cache); print("готово окон", len(cache), flush=True)
rows = []
for t in TYPES:
    for how in ("mean", "max"):
        s = build_scores(tpl, cache, (t, how))
        for r in evaluate(s, gt, ok, official_score, rank_power): rows.append(dict(type=t, agg=how, **r))
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/ag_nem_metrics.csv", index=False); print(tab.round(5).to_string())

# --- шаг 4 D-1: признак AlphaGenome поверх скора v2 (CPU-реранкер: z_v2 + w * z_ag), w и alpha на одной половине теста, оценка на другой
cand = glob.glob("/kaggle/input/**/test_lograte.npy", recursive=True); print(cand)
p2 = np.load([c for c in cand if SP in c][0] if any(SP in c for c in cand) else cand[0]).astype(np.float32).ravel(); assert len(p2) == len(gt)
S = {(t, how): build_scores(tpl, cache, (t, how)) for t in TYPES for how in ("mean",)}
np.savez_compressed(f"{OUT}/ag_nem_scores.npz", **{f"{t}_{h}": np.nan_to_num(v, nan=-1).astype(np.float16) for (t, h), v in S.items()})
cov = ok & np.isfinite(p2) & np.isfinite(S[("CAGE", "mean")]) & np.isfinite(S[("PROCAP", "mean")])
zs = lambda v: (v - v[cov].mean()) / v[cov].std()
z2 = zs(p2); res = []
for t in TYPES:
    za = zs(np.log1p(np.maximum(S[(t, "mean")], 0)))
    for W_ in (0.0, 0.1, 0.2, 0.35, 0.5, 1.0):
        sc = z2 + W_ * za
        for a_ in (0.6, 0.75):
            for hn, hm in (("half1", m1), ("half2", m2)):
                ii = np.flatnonzero(cov & hm)
                if len(ii): res.append(dict(type=t, w=W_, alpha=a_, half=hn, **official_score(rank_power(sc[ii], a_), gt[ii])))
R2 = pd.DataFrame(res); R2.to_csv(f"{OUT}/ag_nem_combine.csv", index=False); print(R2.round(4).to_string())
for t in TYPES:
    out = []
    for ev, tr in (("half1", "half2"), ("half2", "half1")):
        g = R2[(R2.type == t) & (R2.half == tr)]; best = g.sort_values("AUPRC", ascending=False).iloc[0]
        e = R2[(R2.type == t) & (R2.half == ev)]
        base = e[(e.w == 0.0) & (e.alpha == best.alpha)].iloc[0]; pick = e[(e.w == best.w) & (e.alpha == best.alpha)].iloc[0]
        out.append(dict(type=t, eval=ev, w=best.w, alpha=best.alpha, AUPRC_v2=base.AUPRC, AUPRC_comb=pick.AUPRC, Sp_v2=base.Spearman, Sp_comb=pick.Spearman, dAUPRC_pct=(pick.AUPRC / base.AUPRC - 1) * 100))
    print(pd.DataFrame(out).round(4).to_string())
print("СИГНАЛ ЕСТЬ (AUPRC >= 0.01)" if (tab.AUPRC >= 0.01).any() else "СИГНАЛА НЕТ: AUPRC < 0.01 во всех вариантах")
'''
open(f"{root}/run.py", "w", encoding="utf-8").write(head + "\nimport numpy as np\n" + ag + driver)
json.dump({"id": "ivanmitochondrion/epic-alphagenome-nem", "title": "epic-alphagenome-nem", "code_file": "run.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": ["ivanmitochondrion/epic-v2-nematostella"]},
          open(f"{root}/kernel-metadata.json", "w"), indent=1)
print("готово")
