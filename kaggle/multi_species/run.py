# Kaggle-эксперимент: помогает ли клопу (Ofas_2.0) добавить в обучение огнёвку (ilPloInte3.2, тоже насекомое)?
# Признаки для обоих видов считаются ОДИНАКОВО (параметры шага 3б клопа: его PWM старта, k-меры, факторы), модель —
# 61 признак + логотип, деревья глубины 6. Оценка — плотная валидация клопа, официальные метрики.
import glob, json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
BRANCH = "claude/gracious-brown-4ybsmj"
OUT = "/kaggle/working"; DATA = "/tmp/epic"
t0 = time.time()
def log(*a): print(f"[{(time.time() - t0) / 60:5.1f} мин]", *a, flush=True)

for src in glob.glob("/kaggle/input/**/Ofas_2.0.*", recursive=True):
    if not src.endswith((".run.ipynb", "_features.parquet")): shutil.copy(src, OUT)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, "https://github.com/Mitochondrion3000/epic-competiton.git", "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pyarrow"], check=False)
sys.path.insert(0, "/tmp/repo/simple_model")
import features as F, scoring as S
import lightgbm as lgb

# 1) таблица огнёвки — тем же ноутбуком шага 1
env = dict(os.environ, EPIC_SPECIES="ilPloInte3.2", EPIC_DATA_DIR=DATA, EPIC_OUT_DIR=OUT)
r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1",
                    "/tmp/repo/simple_model/step1_data_table.ipynb", "--output", f"{OUT}/ilPloInte3.2.step1.run.ipynb"],
                   cwd="/tmp/repo/simple_model", env=env, capture_output=True, text=True)
if r.returncode: print(r.stderr[-3000:]); sys.exit("шаг 1 для огнёвки упал")
log("шаг 1 огнёвки готов")
env["EPIC_SPECIES"] = "Ofas_2.0"
if not os.path.isdir(f"{DATA}/Ofas_2.0"):
    sys.path.insert(0, "/tmp"); exec(open("/tmp/repo/kaggle/multi_species/download.py").read())

CODE = np.full(256, 4, np.uint8)
for i, b in enumerate("ACGT"): CODE[ord(b)] = CODE[ord(b.lower())] = i
import gzip
def read_genome(path, keep):
    seq, low, name, chunks = {}, {}, None, []
    def flush():
        if name in keep:
            raw = np.frombuffer(b"".join(chunks), np.uint8); seq[name], low[name] = CODE[raw], raw >= ord("a")
    with gzip.open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"): flush(); name, chunks = line[1:].split()[0].decode(), []
            elif name in keep: chunks.append(line.rstrip())
    flush(); return seq, low

P = json.load(open(f"{OUT}/Ofas_2.0.step3b_params.json"))
FE = P["features"]; PARAMS = {"pwm": np.array(P["pwm"]), "kmers": [tuple(k) for k in P["kmers"]], "tf": P["tf"], "wide": True, "local": True}
LOGO = F.logo_names(); ALL = FE + LOGO

def featurize(species, df, extra_contigs=()):
    seq, low = read_genome(f"{DATA}/{species}/genome/genome.fa.gz", set(df.chrom) | set(extra_contigs))   # + контиги плотной валидации
    X = np.zeros((len(df), len(ALL)), np.float32); df = df.reset_index(drop=True)
    for c, d in df.groupby("chrom", sort=False):
        r = d.index.values
        X[r, :len(FE)] = F.feature_matrix(seq[c], low[c], d.pos.values, d.strand.values, PARAMS, FE)
        for st in "+-":
            m = d.strand.values == st
            if m.any(): X[r[m], len(FE):] = F.logo_onehot(seq[c], d.pos.values[m], st)
    return X, df, seq, low

def sample(table, n_pos, n_neg, split):
    t = table[table.split.isin(split)]
    return pd.concat([t[t.y == 1].sample(min(n_pos, int(t.y.sum())), random_state=0), t[t.y == 0].sample(n_neg, random_state=0)])

ofas = pd.read_parquet(f"{OUT}/Ofas_2.0.step1_table.parquet"); meta = json.load(open(f"{OUT}/Ofas_2.0.step1_meta.json"))
plod = pd.read_parquet(f"{OUT}/ilPloInte3.2.step1_table.parquet")
Xo, do, SEQ, LOW = featurize("Ofas_2.0", sample(ofas, 400_000, 1_000_000, ["fit"]), meta["VAL_CONTIGS"]); log("признаки клопа", Xo.shape)
Xp, dp, _, _ = featurize("ilPloInte3.2", sample(plod, 400_000, 1_000_000, ["fit", "val"])); log("признаки огнёвки", Xp.shape)

# плотная валидация клопа (как в шаге 3б): первые половины val-контигов до ~3 млн позиций-цепей (память)
wl = pd.read_csv(f"{DATA}/Ofas_2.0/for_participants/whitelist.train.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], dtype={"chrom": str})
lab = ofas[ofas.y == 1]; dense = []; n = 0
for c in meta["VAL_CONTIGS"]:
    L = len(SEQ[c]) if c in SEQ else 0
    if not L: continue
    inwl = np.zeros(L, bool)
    for s, e in zip(*wl[wl.chrom == c][["start", "end"]].values.T): inwl[s:e] = True
    inwl[L // 2:] = False; dpos = np.flatnonzero(inwl)
    if not len(dpos): continue
    tot = dict(zip(lab[lab.chrom == c].pos.values * 2 + (lab[lab.chrom == c].strand.values == "-"), lab[lab.chrom == c].tot.values))
    for st in "+-":
        X = np.hstack([F.feature_matrix(SEQ[c], LOW[c], dpos, np.full(len(dpos), st), PARAMS, FE), F.logo_onehot(SEQ[c], dpos, st).astype(np.float32)])
        dense.append((X, np.array([tot.get(k, 0.0) for k in (dpos * 2 + (st == "-")).tolist()], np.float32)))
    n += 2 * len(dpos)
    if n >= 3_000_000: break
assert dense, "плотная валидация пуста: val-контиги не загружены"
DY = np.concatenate([d[1] for d in dense]); log("плотная валидация", len(DY), "стартов", int((DY > 0).sum()))

KW = dict(objective="binary", max_depth=6, num_leaves=64, learning_rate=0.05, min_data_in_leaf=200, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, verbose=-1, seed=0)
def run(name, X, y, tot):
    c = lgb.train(KW, lgb.Dataset(X, label=y), 800)
    r = lgb.train({**KW, "objective": "regression"}, lgb.Dataset(X[y == 1], label=np.log1p(tot[y == 1])), 800)
    s = np.concatenate([np.log(np.clip(c.predict(d[0]), 1e-12, 1)) + r.predict(d[0]) for d in dense])
    res = S.official_score(S.rank_power(s, 0.55), DY)
    log(f"{name:28s} AUPRC {res['AUPRC']:.4f}  Spearman {res['Spearman']:.4f}")
    return {"вариант": name, **res}
out = [run("только клоп", Xo, do.y.values, do.tot.values),
       run("клоп + огнёвка", np.vstack([Xo, Xp]), np.r_[do.y.values, dp.y.values], np.r_[do.tot.values, dp.tot.values]),
       run("только огнёвка", Xp, dp.y.values, dp.tot.values)]
pd.DataFrame(out).to_csv(f"{OUT}/multi_species_result.csv", index=False); print(pd.DataFrame(out).to_string())
