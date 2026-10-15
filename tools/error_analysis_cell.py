# Разбор ошибок v2 на hold-out report. Вставить как ячейку ПОСЛЕ ячейки 7.2 (нужны HO, genome, GROUPS, OUT_DIR).
# Результат: OUT_DIR/error_analysis.csv (сводка), error_positions.csv.gz (позиции), печать таблицы.
# Группы: FN = старты (gt>0) вне верхних N по предсказанию; FP = верхние N по предсказанию без старта;
# TP = старты внутри верхних N; BG = случайные не-N позиции. Сравниваем признаки FN/FP с TP и BG.
import numpy as np, pandas as pd
rep = HO["report"]; rows = rep["rows"]; gt = rep["gt"]; p = rep["ens"]; nm = rep["n"]
ok = ~nm; pos = (gt > 0) & ok; N = int(pos.sum())
thr = np.partition(p[ok], -N)[-N]
top = (p >= thr) & ok
grp = np.full(len(gt), "", object)
grp[pos & top] = "TP"; grp[pos & ~top] = "FN"; grp[~pos & top] = "FP"
rng = np.random.default_rng(0)
bg = rng.choice(np.flatnonzero(ok & ~top & ~pos), min(N, int(ok.sum())), replace=False); grp[bg] = "BG"
idx = np.flatnonzero(grp != "")
# координаты каждой позиции: строка -> (контиг, позиция, цепь)
rid = np.searchsorted(rows.off.values, idx, side="right") - 1
chrom = rows.chrom.values[rid]; gpos = rows.start.values[rid] + (idx - rows.off.values[rid]); strand = rows.strand.values[rid]
COMP = np.array([3, 2, 1, 0, 4, 5, 6, 7, 8])
def window(c, g, st, h):  # коды ±h вокруг позиции, в направлении цепи
    s = genome[c]; a = s[max(g - h, 0):g + h + 1]
    a = np.pad(a, (max(h - g, 0), max(g + h + 1 - len(s), 0)), constant_values=8)
    return COMP[a[::-1]] if st else a
recs = []
for c, g, st, i in zip(chrom, gpos, strand, idx):
    w = window(c, int(g), int(st), 100); u = w % 4; low = (w >= 4) & (w < 8); n_ = w == 8
    ok_ = ~n_
    gc = np.isin(u[ok_], [1, 2]).mean() if ok_.any() else np.nan
    cg = np.sum((u[:-1] == 1) & (u[1:] == 2) & ok_[:-1] & ok_[1:])
    c_, g_ = np.sum((u == 1) & ok_), np.sum((u == 2) & ok_)
    k3 = u[:-2] * 16 + u[1:-1] * 4 + u[2:]; cnt = np.bincount(k3[ok_[:-2]], minlength=64) + 0.0
    pr = cnt / max(cnt.sum(), 1); ent = -(pr[pr > 0] * np.log2(pr[pr > 0])).sum()
    c0 = w[100] % 4 if w[100] < 8 else 4; m1 = w[99] % 4 if w[99] < 8 else 4
    recs.append(dict(i=i, grp=grp[i], gt=gt[i], pred=p[i], chrom=c, strand=st, gc=gc, cpg_oe=cg / max(c_ * g_ / max(ok_.sum(), 1), 1e-9),
                     lower=low.mean(), ent3=ent, ref="ACGTN"[c0], prev="ACGTN"[m1], dinuc="ACGTN"[m1] + "ACGTN"[c0]))
df = pd.DataFrame(recs)
# соседние старты в ±50 и локальная плотность старта (по gt того же ряда)
pos_idx = np.flatnonzero(pos)
df["n_starts_pm50"] = [np.sum(np.abs(pos_idx - i) <= 50) - (1 if gt[i] > 0 else 0) for i in df.i]
df["chr_len_rank"] = df.chrom.map(pd.Series(rows.groupby("chrom").len.sum()).rank(ascending=False))
df.to_csv(f"{OUT_DIR}/error_positions.csv.gz", index=False)
num = ["gc", "cpg_oe", "lower", "ent3", "n_starts_pm50", "gt", "pred"]
summ = df.groupby("grp")[num].median().T
summ["доля FN от стартов"] = np.nan; summ.loc["gt", "доля FN от стартов"] = (grp == "FN").sum() / N
from scipy.stats import mannwhitneyu
for g in ("FN", "FP"):
    for f in num[:5]:
        a, b = df[df.grp == g][f].dropna(), df[df.grp == "TP" if g == "FN" else df.grp == "BG"][f].dropna()
        summ.loc[f, f"p({g} vs {'TP' if g=='FN' else 'BG'})"] = mannwhitneyu(a, b).pvalue
summ.to_csv(f"{OUT_DIR}/error_analysis.csv")
print(summ.round(4).to_string())
print("\nдинуклеотид (пред. + позиция старта), доли по группам:")
print(pd.crosstab(df.dinuc, df.grp, normalize="columns").round(3).sort_values("TP", ascending=False).head(10).to_string())
df["gt_bin"] = pd.cut(df["gt"], [-1, 0, 1, 2, 4, 9, 49, 1e9])
print("\nдоля FN среди стартов по силе (чем слабее, тем чаще промах?):")
s = df[df.grp.isin(["FN", "TP"])]; print(s.groupby("gt_bin", observed=True).grp.apply(lambda x: (x == "FN").mean()).round(3).to_string())
print("\nдоля FN по контигам:"); print(s.groupby("chrom").grp.apply(lambda x: (x == "FN").mean()).round(3).head(15).to_string())
