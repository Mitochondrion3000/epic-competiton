# Шаг 7 (эксперимент): интерпретируемая «мотивная» модель для jaNemVect1.1.
# Модель: K фильтров-мотивов (PWM, ширина 15, часть инициализирована известными элементами промотора)
#   -> активация softplus -> для каждого мотива своя гладкая кривая «вклад в зависимости от расстояния до старта»
#   (-150..+50 п.н., по ходу транскрипции) -> сумма. Две головы: шанс старта и сила старта.
# Обучение плотное: куски генома по 1000 п.н., все позиции внутри куска (в т.ч. соседи настоящих стартов).
# Счёт = log P(старт) + E[log1p(счёт) | старт], как в шагах 4–6. Сравнение с шагом 6 на тесте и их смесь.
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn

SMOKE = os.environ.get("SMOKE") == "1"
KAGGLE = os.path.isdir("/kaggle/input")
code = glob.glob("/kaggle/input/**/final.py", recursive=True)
sys.path.insert(0, os.path.dirname(code[0]) if code else "/home/user/epic-competiton/simple_model")
import final as FN, scoring as S

SP = "jaNemVect1.1"
DATA = "/tmp/epic" if KAGGLE else "/home/user/epic_data"
OUT = "/kaggle/working" if KAGGLE else "/tmp/claude-0/-home-user-epic-competiton/7a7d5ba2-1479-5876-9975-58d9838e79f2/scratchpad/motifnet"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0); rng = np.random.default_rng(0)
t0 = time.time()
def log(*a): print(f"[{(time.time() - t0) / 60:6.1f} мин]", *a, flush=True)

K, W, UP, DOWN, L = 32, 15, 150, 50, 1000          # мотивов, ширина, окно кривой (-UP..+DOWN), длина куска
LEFT = W // 2 + UP                                  # сколько букв слева от первой предсказываемой позиции
RIGHT = W - 1 - W // 2 + DOWN
STEPS = 60 if SMOKE else 8000

# ---------- данные ----------
FN.download_dataset(SP, DATA)
D = f"{DATA}/{SP}"
def vec(p): return pd.read_csv(p, header=None, dtype=np.float32, engine="c")[0].values
tr = FN.read_template(f"{D}/for_participants/template.train.bed.gz")
te = FN.read_template(f"{D}/for_participants/template.test.bed.gz")
if SMOKE:
    keep_tr = tr.chrom.drop_duplicates().iloc[:3].tolist(); keep_te = te.chrom.drop_duplicates().iloc[:1].tolist()
else:
    keep_tr, keep_te = tr.chrom.unique().tolist(), te.chrom.unique().tolist()
ytr, yte = vec(f"{D}/for_participants/csRNA-train.txt.gz"), vec(f"{D}/ground_truth/csRNA-test.txt.gz")
assert len(ytr) == tr.len.sum() and len(yte) == te.len.sum()
tr = tr[tr.chrom.isin(keep_tr)]; te = te[te.chrom.isin(keep_te)]
SEQ, _ = FN.read_genome(f"{D}/genome/genome.fa.gz", set(keep_tr) | set(keep_te))
log("геном и разметка прочитаны", len(SEQ), "контигов")

# ориентированные дорожки: для цепи "-" — обратный комплемент, всё развёрнуто (позиция q <-> геномная len-1-q)
COMP = np.array([3, 2, 1, 0, 4], np.uint8)
TR = {}                                             # (контиг, цепь) -> seq, lab (-1 = вне шаблона)
def track(c, s):
    if (c, s) not in TR:
        q = SEQ[c] if s == "+" else COMP[SEQ[c]][::-1].copy()
        TR[c, s] = {"seq": q, "lab": np.full(len(q), -1, np.float32)}
    return TR[c, s]
def fill(t, y):
    for c, s, a, b, o in zip(t.chrom, t.strand, t.start, t.end, t.off):
        tk = track(c, s); v = y[o:o + b - a]; n = len(tk["seq"])
        if s == "+": tk["lab"][a:b] = v
        else: tk["lab"][n - b:n - a] = v[::-1]
fill(tr, ytr); fill(te, yte)

# валидация: часть train-контигов целиком (~10% позиций)
csize = tr.groupby("chrom").len.sum().sample(frac=1, random_state=0)
val_c = set(csize.index[np.cumsum(csize.values) <= max(0.1 * csize.sum(), csize.iloc[0])])
fit_c = set(csize.index) - val_c
log(f"fit-контигов {len(fit_c)}, val-контигов {len(val_c)}")

def intervals(cs):                                  # известные интервалы в ориентированных координатах
    out = []
    for c, s, a, b in zip(tr.chrom, tr.strand, tr.start, tr.end):
        if c in cs:
            n = len(SEQ[c]); out.append(((c, s), a, b) if s == "+" else ((c, s), n - b, n - a))
    return out
def positives(cs):
    return [(k, q) for k in TR if k[0] in cs for q in np.flatnonzero(TR[k]["lab"] > 0)]

ONE = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)])
def window(seq, a, b):                              # seq[a:b] с N за краями
    out = np.full(b - a, 4, np.uint8); lo, hi = max(a, 0), min(b, len(seq))
    if hi > lo: out[lo - a:hi - a] = seq[lo:hi]
    return out

class Sampler:
    def __init__(s, cs, seed):
        s.iv, s.pos, s.r = intervals(cs), positives(cs), np.random.default_rng(seed)
        ln = np.array([b - a for _, a, b in s.iv], float); s.p = ln / ln.sum()
        log(f"сэмплер: интервалов {len(s.iv)}, стартов {len(s.pos)}")
    def batch(s, B=64):
        X = np.zeros((B, 4, L + LEFT + RIGHT), np.float32); Y = np.full((B, L), -1, np.float32)
        for i in range(B):
            if s.r.random() < 0.5:                  # кусок вокруг настоящего старта
                k, q = s.pos[s.r.integers(len(s.pos))]
            else:                                   # случайный кусок из шаблона
                k, a, b = s.iv[s.r.choice(len(s.iv), p=s.p)]; q = s.r.integers(a, b)
            st = q - s.r.integers(L); tk = TR[k]
            X[i] = ONE[window(tk["seq"], st - LEFT, st + L + RIGHT)].T
            Y[i] = window_lab(tk["lab"], st, st + L)
        return torch.from_numpy(X), torch.from_numpy(Y)
def window_lab(lab, a, b):
    out = np.full(b - a, -1, np.float32); lo, hi = max(a, 0), min(b, len(lab))
    if hi > lo: out[lo - a:hi - a] = lab[lo:hi]
    return out

# ---------- модель ----------
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "W": "AT", "S": "CG", "R": "AG", "Y": "CT", "K": "GT", "M": "AC", "V": "ACG", "N": "ACGT"}
SEEDS = {"TATA": "TATAWAWR", "GC-бокс": "GGGCGG", "CCAAT": "CCAAT", "DPE": "RGWYV", "MTE": "CSARCSSAAC", "E-бокс": "CACGTG"}
INR = np.array([[-0.1, -0.13, 0.05, -0.26, -0.71, 0.73, 0.08, 0.02, -0.01, 0., -0.05],      # PWM Inr из шага 3б (A,C,G,T; -5..+5)
                [0.1, 0.27, -0.18, 0.04, 0.68, -0.86, -0.47, -0.39, -0.28, -0.19, -0.18],
                [0.13, 0.15, 0.32, 0.19, -0.21, 0.25, 0.05, 0.28, 0.35, 0.27, 0.34],
                [-0.07, -0.21, -0.2, 0.08, 0.09, -0.94, 0.15, 0., -0.1, -0.08, -0.11]], np.float32)

class MotifNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.motif = nn.Conv1d(4, K, W)
        s.curve = nn.Conv1d(K, 2 * K, UP + DOWN + 1, groups=K, bias=False)   # у мотива k кривые 2k (шанс) и 2k+1 (сила)
        s.bias = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            s.motif.weight.mul_(0.3); s.curve.weight.normal_(0, 1e-3)
            def put(k, m):                          # m: 4×w, кладём в центр фильтра
                w0 = (W - m.shape[1]) // 2; s.motif.weight[k].zero_(); s.motif.weight[k, :, w0:w0 + m.shape[1]] = torch.tensor(m)
            put(0, 2 * INR)
            for k, cons in enumerate(SEEDS.values(), 1):
                put(k, np.array([[1.0 if b in IUPAC[ch] else -1.0 for ch in cons] for b in "ACGT"], np.float32))
            s.motif.bias[:len(SEEDS) + 1] = -2.0
    def forward(s, x):
        z = s.curve(Fn.softplus(s.motif(x)))
        return z.view(z.shape[0], K, 2, -1).sum(1) + s.bias[None, :, None]
    def roughness(s):
        w = s.curve.weight[:, 0]
        return ((w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]) ** 2).sum()
NAMES = ["Inr"] + list(SEEDS) + [f"m{k}" for k in range(len(SEEDS) + 1, K)]

def loss_fn(net, X, Y, lam=30.0):
    z = net(X); m = Y >= 0; y = (Y > 0)
    bce = Fn.binary_cross_entropy_with_logits(z[:, 0][m], y[m].float())
    mse = Fn.mse_loss(z[:, 1][y], torch.log1p(Y[y])) if y.any() else z.sum() * 0
    return bce, mse, bce + 0.05 * mse + lam * net.roughness()

# ---------- обучение ----------
fit_s, val_s = Sampler(fit_c, 1), Sampler(val_c, 2)
VB = [val_s.batch() for _ in range(4 if SMOKE else 25)]
net = MotifNet(); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS, pct_start=0.05)
best, best_state = 1e9, None
for step in range(1, STEPS + 1):
    X, Y = fit_s.batch(); bce, mse, tot = loss_fn(net, X, Y)
    opt.zero_grad(); tot.backward(); opt.step(); sched.step()
    if step % (20 if SMOKE else 500) == 0:
        with torch.no_grad():
            vb = np.mean([[t.item() for t in loss_fn(net, x, y)[:2]] for x, y in VB], axis=0)
        v = vb[0] + 0.05 * vb[1]
        if v < best: best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        log(f"шаг {step}: train bce {bce.item():.4f} mse {mse.item():.3f} | val bce {vb[0]:.4f} mse {vb[1]:.3f}")
net.load_state_dict(best_state); net.eval()
torch.save(net.state_dict(), f"{OUT}/{SP}.motifnet.pt")

# ---------- плотное предсказание ----------
@torch.no_grad()
def predict_track(k, seg=1_000_000):
    seq = TR[k]["seq"]; out = np.empty(len(seq), np.float32)
    for a in range(0, len(seq), seg):
        b = min(a + seg, len(seq))
        x = torch.from_numpy(ONE[window(seq, a - LEFT, b + RIGHT)].T[None].copy())
        z = net(x)[0]; out[a:b] = (Fn.logsigmoid(z[0]) + z[1]).numpy()
    out[seq == 4] = -np.inf
    return out
def template_scores(t):
    sc, cache = np.empty(int(t.len.sum()), np.float32), {}
    for c, s, a, b, o in zip(t.chrom, t.strand, t.start, t.end, t.off):
        if (c, s) not in cache: cache.clear(); cache[c, s] = predict_track((c, s))
        p = cache[c, s]; n = len(p)
        sc[o:o + b - a] = p[a:b] if s == "+" else p[n - b:n - a][::-1]
    return sc
def offsets(t): return t.assign(off=np.r_[0, np.cumsum(t.len.values)[:-1]])

tv = offsets(tr[tr.chrom.isin(val_c)].sort_values(["chrom", "strand"], kind="stable"))
yv = np.concatenate([TR[c, s]["lab"][a:b] if s == "+" else TR[c, s]["lab"][len(SEQ[c]) - b:len(SEQ[c]) - a][::-1]
                     for c, s, a, b in zip(tv.chrom, tv.strand, tv.start, tv.end)])
sv = template_scores(tv); nv = ~np.isfinite(sv)
res = {"val": {}, "test": {}}
for al in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    res["val"][al] = S.official_score(S.rank_power(sv, al, nv), yv); log("val alpha", al, res["val"][al])
mx = max(r["AUPRC"] for r in res["val"].values())
alpha = max((a for a, r in res["val"].items() if r["AUPRC"] >= 0.995 * mx), key=lambda a: res["val"][a]["Spearman"])
log("выбран alpha", alpha)

te_full = FN.read_template(f"{D}/for_participants/template.test.bed.gz")
sel = te_full.sort_values(["chrom", "strand"], kind="stable"); sel = sel[sel.chrom.isin(keep_te)]   # по контигам -> в порядок шаблона
dst = np.concatenate([np.arange(o, o + n) for o, n in zip(sel.off.values, sel.len.values)])
score = np.full(len(yte), -np.inf, np.float32); score[dst] = template_scores(offsets(sel))
mask = np.zeros(len(yte), bool); mask[dst] = True; nmask = ~np.isfinite(score)
res["test"]["motifnet"] = S.official_score(S.rank_power(score, alpha, nmask)[mask], yte[mask]); log("тест motifnet", res["test"]["motifnet"])

p6f = glob.glob("/kaggle/input/**/jaNemVect1.1.simple_model_final.predictions.txt.gz", recursive=True) or \
      glob.glob("/home/user/epic_out/kaggle_nem6/jaNemVect1.1.simple_model_final.predictions.txt.gz")
if p6f:
    from scipy.stats import rankdata
    p6 = vec(p6f[0]); r6 = rankdata(p6[mask]) / mask.sum(); rn = rankdata(np.where(nmask, -1e9, score)[mask]) / mask.sum()
    for w in (0.0, 0.3, 0.5, 0.7):
        res["test"][f"смесь w_motifnet={w}"] = S.official_score(S.rank_power(w * rn + (1 - w) * r6, alpha, nmask[mask]), yte[mask])
        log(f"тест смесь w={w}", res["test"][f"смесь w_motifnet={w}"])

# ---------- интерпретация ----------
with torch.no_grad():
    contrib = []                                    # важность мотива: разброс его вклада в шанс старта на val-кусках
    for x, _ in VB[:8]:
        a = Fn.softplus(net.motif(x)); z = net.curve(a)[:, 0::2]; contrib.append(z.std(dim=(0, 2)).numpy())
imp = np.mean(contrib, 0)
Wm, Wc = net.motif.weight.detach().numpy(), net.curve.weight[:, 0].detach().numpy()
d = np.arange(-UP, DOWN + 1)
info = []
for k in np.argsort(-imp):
    m = Wm[k] - Wm[k].mean(0); cons = "".join("ACGT"[j] if m[j, i] > 0.5 * np.abs(m).max() else "n" for i, j in enumerate(m.argmax(0)))
    cv = Wc[2 * k]; info.append({"мотив": NAMES[k], "важность": float(imp[k]), "консенсус": cons.strip("n") or "-",
                                 "пик кривой, п.н.": int(d[np.argmax(np.abs(cv))]), "знак": "+" if cv[np.argmax(np.abs(cv))] > 0 else "-"})
res["мотивы"] = info; res["alpha"] = alpha
json.dump(res, open(f"{OUT}/{SP}.motifnet_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
np.savez(f"{OUT}/{SP}.motifnet_weights.npz", motif=Wm, motif_bias=net.motif.bias.detach().numpy(), curve=net.curve.weight.detach().numpy()[:, 0], names=NAMES, importance=imp)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = np.argsort(-imp)[:12]; fig, ax = plt.subplots(12, 2, figsize=(11, 22), gridspec_kw={"width_ratios": [1, 3]})
    for r, k in enumerate(top):
        m = Wm[k] - Wm[k].mean(0); ax[r, 0].imshow(m, cmap="RdBu_r", vmin=-np.abs(m).max(), vmax=np.abs(m).max(), aspect="auto")
        ax[r, 0].set_yticks(range(4)); ax[r, 0].set_yticklabels("ACGT"); ax[r, 0].set_xticks([]); ax[r, 0].set_title(f"{NAMES[k]} ({info[r]['консенсус']})", fontsize=8)
        ax[r, 1].plot(d, Wc[2 * k], label="шанс"); ax[r, 1].plot(d, Wc[2 * k + 1], label="сила", alpha=0.6); ax[r, 1].axvline(0, c="k", lw=0.5)
        ax[r, 1].axhline(0, c="k", lw=0.3); ax[r, 1].tick_params(labelsize=7)
    ax[0, 1].legend(fontsize=7); ax[-1, 1].set_xlabel("положение мотива относительно старта, п.н.")
    plt.tight_layout(); plt.savefig(f"{OUT}/{SP}.motifnet_motifs.png", dpi=110)
except Exception as e: log("рисунок не построен:", e)
log("ГОТОВО", json.dumps(res["test"], ensure_ascii=False, default=str))
