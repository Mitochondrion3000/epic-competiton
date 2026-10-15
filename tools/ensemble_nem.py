# Совпадают ли ошибки у v2 и простой модели на Nematostella (тест с метками) и даёт ли смесь прирост. CPU, готовые скоры.
import glob
v2f = glob.glob("/kaggle/input/**/jaNemVect1.1/test_lograte.npy", recursive=True) or glob.glob("/kaggle/input/**/test_lograte.npy", recursive=True)
smf = glob.glob("/kaggle/input/**/jaNemVect1.1.step6_score.npy", recursive=True)
print(v2f, smf)
a = np.load(v2f[0]).astype(np.float32); b = np.load(smf[0]).astype(np.float32)
assert len(a) == len(b) == len(gt), (len(a), len(b), len(gt))
ok = (m1 | m2) & np.isfinite(a) & np.isfinite(b)
za = (a - a[ok].mean()) / a[ok].std(); zb = (b - b[ok].mean()) / b[ok].std(); za[~ok] = za[ok].min(); zb[~ok] = zb[ok].min()
pos = (gt > 0) & ok; N = int(pos.sum())
R = {"N_starts": N, "N_scored": int(ok.sum())}
for nm, z in (("v2", za), ("simple", zb)): R[f"metric_{nm}"] = {h: official_score(rank_power(z[mk & ok], 0.6) if False else rank_power(z[ok & mk], 0.6), gt[ok & mk]) for h, mk in (("half1", m1), ("half2", m2))}
# корреляция скоров: по всем оцениваемым (выборка) и среди стартов
rng = np.random.default_rng(0); s = rng.choice(np.flatnonzero(ok), 2_000_000, replace=False)
rk = lambda x: np.argsort(np.argsort(x)).astype(np.float64)
R["spearman_all"] = float(np.corrcoef(rk(za[s]), rk(zb[s]))[0, 1]); ip = np.flatnonzero(pos)
R["spearman_on_starts"] = float(np.corrcoef(rk(za[ip]), rk(zb[ip]))[0, 1])
# пересечение верхних N
def topn(z): thr = np.partition(z[ok], -N)[-N]; return ok & (z >= thr)
ta, tb = topn(za), topn(zb)
tp_a, tp_b = ta & pos, tb & pos
R["top_overlap"] = dict(top_a=int(ta.sum()), top_b=int(tb.sum()), both=int((ta & tb).sum()), TP_a=int(tp_a.sum()), TP_b=int(tp_b.sum()), TP_both=int((tp_a & tp_b).sum()), TP_union=int((tp_a | tp_b).sum()),
                       FP_a=int((ta & ~pos).sum()), FP_b=int((tb & ~pos).sum()), FP_both=int((ta & tb & ~pos).sum()), FN_both=int((pos & ~ta & ~tb).sum()))
# смеси: вес w на v2, alpha 0.6/0.75; выбор на одной половине, оценка на другой
rows = []
for w in np.round(np.linspace(0, 1, 11), 2):
    zz = (w * za + (1 - w) * zb).astype(np.float32)
    for al in (0.6, 0.75):
        for hn, mk in (("half1", m1), ("half2", m2)):
            idx = np.flatnonzero(ok & mk); rows.append(dict(w_v2=w, alpha=al, half=hn, **official_score(rank_power(zz[idx], al), gt[idx])))
tab = pd.DataFrame(rows); tab.to_csv("/kaggle/working/ensemble_nem_grid.csv", index=False)
res = []
for ev, tr in (("half1", "half2"), ("half2", "half1")):
    g = tab[tab.half == tr].sort_values("AUPRC", ascending=False).iloc[0]
    p = tab[(tab.half == ev) & (tab.w_v2 == g.w_v2) & (tab.alpha == g.alpha)].iloc[0]
    b0 = tab[(tab.half == ev) & (tab.w_v2 == 1.0)].sort_values("AUPRC", ascending=False).iloc[0]
    res.append(dict(eval=ev, w_v2=g.w_v2, alpha=g.alpha, AUPRC=p.AUPRC, Spearman=p.Spearman, v2_only_AUPRC=b0.AUPRC, v2_only_Spearman=b0.Spearman))
R["blend_cv"] = res
print(json.dumps(R, indent=1, ensure_ascii=False, default=float)); print(tab.pivot_table(index="w_v2", columns=["alpha", "half"], values="AUPRC").round(4).to_string())
print(tab.pivot_table(index="w_v2", columns=["alpha", "half"], values="Spearman").round(4).to_string())
json.dump(R, open("/kaggle/working/ensemble_nem.json", "w"), ensure_ascii=False, default=float)
