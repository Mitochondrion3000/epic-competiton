# Kaggle-скрипт: эксперимент «веса трудных примеров» для простой модели (Nematostella, плотная валидация, CPU).
# Берёт ячейки 0-7 из simple_model/step6_final.ipynb (признаки, модель на fit, плотная валидация) и добавляет сравнение вариантов весов классификатора.
import glob, json, os, shutil, subprocess, sys, time
BRANCH = "main"
REPO = "https://github.com/Mitochondrion3000/epic-competiton.git"
SPECIES = "jaNemVect1.1"
for src in glob.glob(f"/kaggle/input/**/{SPECIES}.*", recursive=True):
    if not src.endswith((".run.ipynb", "_features.parquet")):
        shutil.copy(src, "/kaggle/working/"); print("взят вход:", os.path.basename(src))
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pyarrow"], check=False)
CELL = r'''
# --- эксперимент: веса трудных примеров для классификатора шанса старта
t_exp = time.time()
cls0, reg0 = MODEL_FIT[0]
ys, tots = rows.y.values, rows.tot.values
neg = fit_idx[ys[fit_idx] == 0]
sc = np.concatenate([cls0.predict(np.hstack([XB[neg[a:a + 200_000]], XE[neg[a:a + 200_000]]]).astype(np.float64)) for a in range(0, len(neg), 200_000)])
hard = neg[sc >= np.quantile(sc, 0.99)]
weak = np.flatnonzero((ys == 1) & (tots <= 2))
iso = np.zeros(len(rows), bool)
P = rows[rows.y == 1][["chrom", "strand", "pos"]]
for (c, st), g in P.groupby(["chrom", "strand"]):
    o = np.argsort(g.pos.values); ps = g.pos.values[o]
    cnt = np.searchsorted(ps, ps + 50, "right") - np.searchsorted(ps, ps - 50, "left") - 1
    iso[g.index.values[o]] = cnt == 0
print(f"трудные негативы {len(hard):,}, слабые позитивы {len(weak):,}, одиночные позитивы {int(iso.sum()):,}", flush=True)
def W(weak_w=1.0, hard_w=1.0, iso_w=1.0):
    w = np.ones(len(rows), np.float64); w[weak] *= weak_w; w[hard] *= hard_w; w[np.flatnonzero(iso)] *= iso_w; return w
VARIANTS = {"базовая": W(), "слабые x3": W(weak_w=3), "трудные негативы x3": W(hard_w=3), "слабые+трудные x3": W(weak_w=3, hard_w=3), "одиночные x3": W(iso_w=3), "одиночные+трудные x3": W(iso_w=3, hard_w=3)}
CLS = {"базовая": cls0}
for name, w in VARIANTS.items():
    if name in CLS: continue
    CLS[name] = lgb.train({**LGB, "objective": "binary", "seed": 0}, lgb.Dataset(Rows(fit_idx), label=ys[fit_idx], weight=w[fit_idx], feature_name=ALL, free_raw_data=True), N_TREES)
    print(f"обучен вариант «{name}» ({(time.time() - t_exp) / 60:.1f} мин)", flush=True)
out = {k: [] for k in CLS}; ysd = []; n_dense = 0
for c in meta["VAL_CONTIGS"]:
    L = len(SEQ[c]); inwl = np.zeros(L, bool)
    for s, e in zip(*wl_train[wl_train.chrom == c][["start", "end"]].values.T): inwl[s:e] = True
    inwl[L // 2:] = False
    dpos = np.flatnonzero(inwl)
    if len(dpos) == 0: continue
    lab = lab_all[lab_all.chrom == c]
    tot = dict(zip(lab.pos.values * 2 + (lab.strand.values == "-"), lab.tot.values))
    for st in ("+", "-"):
        base = F.feature_matrix(SEQ[c], LOW[c], dpos, np.full(len(dpos), st), PARAMS, BASE_FEATURES)
        res = {k: np.zeros(len(dpos), np.float32) for k in CLS}
        for a in range(0, len(dpos), 1_000_000):
            b = min(a + 1_000_000, len(dpos))
            Xc = np.hstack([base[a:b], extra(SEQ[c], dpos[a:b], st).astype(np.float32)])
            r = reg0.predict(Xc)
            for k, m in CLS.items(): res[k][a:b] = np.log(np.clip(m.predict(Xc), 1e-12, 1)) + r
        for k in CLS: out[k].append(res[k])
        ysd.append(np.array([tot.get(k, 0.0) for k in (dpos * 2 + (st == "-")).tolist()], np.float32))
    n_dense += 2 * len(dpos)
    if n_dense >= 12_000_000: break
Y = np.concatenate(ysd); rows_out = []
for k in CLS:
    Sx = np.concatenate(out[k])
    for a in (0.5, 0.6, 0.7): rows_out.append({"вариант": k, "alpha": a, **S.official_score(S.rank_power(Sx, a), Y)})
tab = pd.DataFrame(rows_out); tab.to_csv(f"{OUT_DIR}/weights_experiment.csv", index=False)
print(f"плотная валидация: {len(Y):,} позиций, стартов {(Y > 0).sum():,}")
print(tab.pivot(index="вариант", columns="alpha", values="AUPRC").round(4).to_string()); print(tab.pivot(index="вариант", columns="alpha", values="Spearman").round(4).to_string())
'''
nb = json.load(open("/tmp/repo/simple_model/step6_final.ipynb"))
cells = nb["cells"][:8]
cells[7]["source"] = [l for l in cells[7]["source"] if "del MODEL_FIT" not in l]
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CELL.splitlines(True)})
nb["cells"] = cells
json.dump(nb, open("/tmp/repo/simple_model/step6_weights.ipynb", "w"))
env = dict(os.environ, EPIC_SPECIES=SPECIES, EPIC_DATA_DIR="/tmp/epic", EPIC_OUT_DIR="/kaggle/working")
r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1", "/tmp/repo/simple_model/step6_weights.ipynb", "--output", f"/kaggle/working/{SPECIES}.step6_weights.run.ipynb"],
                   cwd="/tmp/repo/simple_model", env=env, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
if r.returncode != 0: sys.exit("упал")
