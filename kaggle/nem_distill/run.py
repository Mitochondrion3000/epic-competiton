# I-2: дистилляция v2 в интерпретируемую мотивную модель на Nematostella (jaNemVect1.1).
# Цель = плотный лог-рейт v2 на тестовых контигах (`ivanmitochondrion/epic-v2-nematostella`, test_lograte.npy), метки при обучении НЕ используются.
# Контиги теста делятся на две половины A и B (по размеру); учим на одной (FOLD=0: A), проверяем на другой (B) по Spearman/Pearson с v2
# и по настоящим меткам (AUPRC, Spearman, официальный скорер). Модель: K компонент "PWM -> softplus -> своя кривая влияния по расстоянию"
# + острый Инициатор (свободный фильтр -CORE..+CORE, без сглаживания) + смещение, сумма в лог-пространстве.
# Окружение: EPIC_K (число компонент), EPIC_FOLD (0/1), EPIC_STEPS, SMOKE=1 (быстрый прогон).
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn

SMOKE = os.environ.get("SMOKE") == "1"
KAGGLE = os.path.isdir("/kaggle/input")
code = glob.glob("/kaggle/input/**/final.py", recursive=True)
sys.path.insert(0, os.path.dirname(code[0]) if code else "/home/user/epic-competiton/simple_model")
import final as FN, scoring as S

SP = "jaNemVect1.1"
K = int(os.environ.get("EPIC_K", "32")); FOLD = int(os.environ.get("EPIC_FOLD", "0"))
STEPS = int(os.environ.get("EPIC_STEPS", "60" if SMOKE else "8000"))
TAG = f"k{K}_f{FOLD}" + os.environ.get("EPIC_TAGX", "")
DATA = "/tmp/epic" if KAGGLE else "/home/user/epic_data"
OUT = "/kaggle/working" if KAGGLE else "/tmp/distill_out"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 4); torch.manual_seed(FOLD); rng = np.random.default_rng(FOLD)
t0 = time.time()
def log(*a): print(f"[{(time.time() - t0) / 60:6.1f} мин]", *a, flush=True)
log("устройство", DEV, "| K", K, "| fold", FOLD, "| шагов", STEPS)

W, UP, DOWN, L, CORE = 15, 300, 100, 1000, 10
LEFT = W // 2 + UP
RIGHT = W - 1 - W // 2 + DOWN

# ---------- данные ----------
FN.download_dataset(SP, DATA)
D = f"{DATA}/{SP}"
def vec(p): return pd.read_csv(p, header=None, dtype=np.float32, engine="c")[0].values
te = FN.read_template(f"{D}/for_participants/template.test.bed.gz")
yte = vec(f"{D}/ground_truth/csRNA-test.txt.gz")
v2p = glob.glob("/kaggle/input/**/jaNemVect1.1/test_lograte.npy", recursive=True) or glob.glob("/kaggle/input/**/test_lograte.npy", recursive=True)
assert v2p, "нет test_lograte.npy v2 (источник ядра ivanmitochondrion/epic-v2-nematostella)"
v2 = np.load(v2p[0]).astype(np.float32)
assert len(yte) == len(v2) == te.len.sum(), (len(yte), len(v2), te.len.sum())
log("v2 лог-рейт:", v2p[0], "| конечных", int(np.isfinite(v2).sum()), "из", len(v2))
chroms = te.chrom.unique().tolist()
if SMOKE: te = te[te.chrom.isin(chroms[:2])]
SEQ, _ = FN.read_genome(f"{D}/genome/genome.fa.gz", set(te.chrom))
csize = te.groupby("chrom").len.sum().sample(frac=1, random_state=0)
cum = np.cumsum(csize.values); half = cum <= cum[-1] / 2
if half.all() or not half.any(): half = np.arange(len(csize)) < max(1, len(csize) // 2)
A = set(csize.index[half]); B = set(csize.index) - A
tr_c, ev_c = (A, B) if FOLD == 0 else (B, A)
log(f"контигов обучение {len(tr_c)} ({csize[list(tr_c)].sum() / 1e6:.1f} Мнт), проверка {len(ev_c)} ({csize[list(ev_c)].sum() / 1e6:.1f} Мнт)")

COMP = np.array([3, 2, 1, 0, 4], np.uint8)
TR = {}
def track(c, s):
    if (c, s) not in TR:
        q = SEQ[c] if s == "+" else COMP[SEQ[c]][::-1].copy()
        TR[c, s] = {"seq": q, "tgt": np.full(len(q), np.nan, np.float32)}
    return TR[c, s]
for c, s, a, b, o in zip(te.chrom, te.strand, te.start, te.end, te.off):
    tk = track(c, s); v = v2[o:o + b - a]; n = len(tk["seq"])
    v = np.where(np.isfinite(v), v, np.nan)
    if s == "+": tk["tgt"][a:b] = v
    else: tk["tgt"][n - b:n - a] = v[::-1]
fin = np.concatenate([TR[k]["tgt"][np.isfinite(TR[k]["tgt"])] for k in TR if k[0] in tr_c])
MU, SD = float(fin.mean()), float(fin.std()); Q = float(np.quantile(fin, 0.995)); Q95 = float(np.quantile(fin, 0.95)); del fin
PEAKW = float(os.environ.get("EPIC_PEAKW", "3")); TOPFR = float(os.environ.get("EPIC_TOPFR", "0.4"))
log(f"цель: среднее {MU:.3f}, sd {SD:.3f}, q99.5 {Q:.3f}")

def intervals(cs):
    return [((c, s), a, b) if s == "+" else ((c, s), len(SEQ[c]) - b, len(SEQ[c]) - a)
            for c, s, a, b in zip(te.chrom, te.strand, te.start, te.end) if c in cs]
ONE = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)])
def window(arr, a, b, fill):
    out = np.full(b - a, fill, arr.dtype); lo, hi = max(a, 0), min(b, len(arr))
    if hi > lo: out[lo - a:hi - a] = arr[lo:hi]
    return out

class Sampler:
    def __init__(s, cs, seed):
        s.iv, s.r = intervals(cs), np.random.default_rng(seed)
        ln = np.array([b - a for _, a, b in s.iv], float); s.p = ln / ln.sum()
        s.top = []                                  # места, где v2 особенно уверена: берём куски вокруг них
        for k in {k for k, _, _ in s.iv}:
            t = TR[k]["tgt"]; idx = np.flatnonzero(t > Q)
            s.top += [(k, q) for q in idx[s.r.permutation(len(idx))[:20000]]]
        log(f"сэмплер: интервалов {len(s.iv)}, верхних позиций v2 {len(s.top)}")
    def batch(s, B=64):
        X = np.zeros((B, 4, L + LEFT + RIGHT), np.float32); Y = np.full((B, L), np.nan, np.float32)
        for i in range(B):
            if s.top and s.r.random() < TOPFR: k, q = s.top[s.r.integers(len(s.top))]
            else:
                k, a, b = s.iv[s.r.choice(len(s.iv), p=s.p)]; q = s.r.integers(a, b)
            st = q - s.r.integers(L); tk = TR[k]
            X[i] = ONE[window(tk["seq"], st - LEFT, st + L + RIGHT, 4)].T
            Y[i] = window(tk["tgt"], st, st + L, np.nan)
        return torch.from_numpy(X), torch.from_numpy(Y)

# ---------- модель ----------
INR = np.array([[-0.1, -0.13, 0.05, -0.26, -0.71, 0.73, 0.08, 0.02, -0.01, 0., -0.05],
                [0.1, 0.27, -0.18, 0.04, 0.68, -0.86, -0.47, -0.39, -0.28, -0.19, -0.18],
                [0.13, 0.15, 0.32, 0.19, -0.21, 0.25, 0.05, 0.28, 0.35, 0.27, 0.34],
                [-0.07, -0.21, -0.2, 0.08, 0.09, -0.94, 0.15, 0., -0.1, -0.08, -0.11]], np.float32)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.motif = nn.Conv1d(4, K, W)
        s.curve = nn.Conv1d(K, K, UP + DOWN + 1, groups=K, bias=False)
        s.core = nn.Conv1d(4, 1, 2 * CORE + 1, bias=False)        # острый Инициатор: свободный вес каждой буквы в -CORE..+CORE
        s.bias = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            s.motif.weight.mul_(0.3); s.curve.weight.normal_(0, 1e-3)
            s.core.weight[0, :, :INR.shape[1]] = torch.tensor(INR) * 0.3
    def forward(s, x):
        z = s.curve(Fn.softplus(s.motif(x))); n = z.shape[-1]
        return z.sum(1) + s.core(x)[:, 0, LEFT - CORE:LEFT - CORE + n] + s.bias
    def roughness(s):
        w = s.curve.weight[:, 0]; r = (w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]) ** 2
        r[:, UP - 1 - 5:UP + 5] = 0                 # около старта (|d| ≤ 5) острые пики разрешены
        return r.sum()
net = Net().to(DEV)

def loss_fn(X, Y, lam=0.05):
    z = net(X.to(DEV)); y = ((Y - MU) / SD).to(DEV); m = torch.isfinite(y)
    Yd = Y.to(DEV); wt = 1.0 + PEAKW * (Yd > Q).float() + PEAKW / 3 * (Yd > Q95).float()
    mse = (((z - torch.nan_to_num(y)) ** 2) * wt)[m].sum() / wt[m].sum()
    return mse, mse + lam * net.roughness()

# ---------- обучение ----------
fit_s = Sampler(tr_c, 1); VB = [Sampler(ev_c, 2).batch() for _ in range(4 if SMOKE else 25)]
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS, pct_start=0.05)
best, best_state = 1e9, None
for step in range(1, STEPS + 1):
    X, Y = fit_s.batch(); mse, tot = loss_fn(X, Y)
    opt.zero_grad(); tot.backward(); opt.step(); sched.step()
    if step % (20 if SMOKE else 500) == 0:
        with torch.no_grad(): v = float(np.mean([loss_fn(x, y)[0].item() for x, y in VB]))
        if v < best: best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
        log(f"шаг {step}: train mse {mse.item():.4f} | проверка mse {v:.4f} (1 - R² ≈ {v:.3f})")
net.load_state_dict(best_state); net.eval()
torch.save(net.state_dict(), f"{OUT}/{SP}.distill_{TAG}.pt")

# ---------- плотное предсказание и оценка на проверочных контигах ----------
@torch.no_grad()
def predict_track(k, seg=500_000):
    seq = TR[k]["seq"]; out = np.empty(len(seq), np.float32)
    for a in range(0, len(seq), seg):
        b = min(a + seg, len(seq))
        x = torch.from_numpy(ONE[window(seq, a - LEFT, b + RIGHT, 4)].T[None].copy()).to(DEV)
        out[a:b] = net(x)[0].cpu().numpy() * SD + MU
    out[seq == 4] = -np.inf
    return out
sel = te[te.chrom.isin(ev_c)].sort_values(["chrom", "strand"], kind="stable")
dst = np.concatenate([np.arange(o, o + n) for o, n in zip(sel.off.values, sel.len.values)])
score = np.full(len(yte), -np.inf, np.float32); cache = {}
for c, s, a, b, o in zip(sel.chrom, sel.strand, sel.start, sel.end, sel.off):
    if (c, s) not in cache: cache.clear(); cache[c, s] = predict_track((c, s))
    p = cache[c, s]; n = len(p); score[o:o + b - a] = p[a:b] if s == "+" else p[n - b:n - a][::-1]
mask = np.zeros(len(yte), bool); mask[dst] = True
ref = np.where(np.isfinite(v2), v2, -np.inf).astype(np.float32); ref[~mask] = -np.inf
ok = mask & np.isfinite(score) & np.isfinite(v2)
sub = rng.choice(np.flatnonzero(ok), min(3_000_000, int(ok.sum())), replace=False)
from scipy.stats import spearmanr, pearsonr
res = {"K": K, "fold": FOLD, "mse_val": best,
       "spearman_c_v2": float(spearmanr(score[sub], v2[sub])[0]), "pearson_c_v2": float(pearsonr(score[sub], v2[sub])[0])}
top = ok & (v2 > np.quantile(v2[ok], 0.99)); sub2 = np.flatnonzero(top)[:2_000_000]
res["spearman_top1pct_v2"] = float(spearmanr(score[sub2], v2[sub2])[0])
log("согласие с v2 на проверочных контигах:", {k: round(v, 3) for k, v in res.items() if k.startswith(("spear", "pears"))})
res["labels"] = {}
nm_s, nm_r = ~np.isfinite(score), ~np.isfinite(ref)
for al in (0.3, 0.5, 0.7, 1.0):
    res["labels"][f"distill_a{al}"] = S.official_score(S.rank_power(score, al, nm_s)[mask], yte[mask])
    res["labels"][f"v2_a{al}"] = S.official_score(S.rank_power(ref, al, nm_r)[mask], yte[mask])
    log("alpha", al, "дистилляция", res["labels"][f"distill_a{al}"], "| v2", res["labels"][f"v2_a{al}"])

# ---------- интерпретация ----------
with torch.no_grad():
    imp = np.mean([net.curve(Fn.softplus(net.motif(x.to(DEV)))).std(dim=(0, 2)).cpu().numpy() for x, _ in VB[:8]], 0)
Wm, Wc = net.motif.weight.detach().cpu().numpy(), net.curve.weight[:, 0].detach().cpu().numpy()
d = np.arange(-UP, DOWN + 1); info = []
for k in np.argsort(-imp):
    m = Wm[k] - Wm[k].mean(0); cons = "".join("ACGT"[j] if m[j, i] > 0.5 * np.abs(m).max() else "n" for i, j in enumerate(m.argmax(0)))
    cv = Wc[k]; info.append({"компонента": int(k), "важность": float(imp[k]), "консенсус": cons.strip("n") or "-",
                             "пик кривой, п.н.": int(d[np.argmax(np.abs(cv))]), "знак": "+" if cv[np.argmax(np.abs(cv))] > 0 else "-"})
res["компоненты"] = info
json.dump(res, open(f"{OUT}/{SP}.distill_{TAG}_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
np.savez(f"{OUT}/{SP}.distill_{TAG}_weights.npz", core=net.core.weight.detach().cpu().numpy(), motif=Wm, motif_bias=net.motif.bias.detach().cpu().numpy(),
         curve=Wc, importance=imp, mu_sd=np.array([MU, SD]))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nt = min(12, K); order = np.argsort(-imp)[:nt]; fig, ax = plt.subplots(nt, 2, figsize=(11, 1.9 * nt), gridspec_kw={"width_ratios": [1, 3]})
    for r, k in enumerate(order):
        m = Wm[k] - Wm[k].mean(0); ax[r, 0].imshow(m, cmap="RdBu_r", vmin=-np.abs(m).max(), vmax=np.abs(m).max(), aspect="auto")
        ax[r, 0].set_yticks(range(4)); ax[r, 0].set_yticklabels("ACGT"); ax[r, 0].set_xticks([]); ax[r, 0].set_title(f"#{k} ({info[r]['консенсус']})", fontsize=8)
        ax[r, 1].plot(d, Wc[k]); ax[r, 1].axvline(0, c="k", lw=0.5); ax[r, 1].axhline(0, c="k", lw=0.3); ax[r, 1].tick_params(labelsize=7)
    ax[-1, 1].set_xlabel("положение мотива относительно старта, п.н."); plt.tight_layout(); plt.savefig(f"{OUT}/{SP}.distill_{TAG}_components.png", dpi=110)
except Exception as e: log("рисунок не построен:", e)
log("ГОТОВО")
