import glob, gzip, os, numpy as np
SPECIES = "sSquSuc5.hap2"
SCALE = 100_000
OUT = "/kaggle/working/m3ft_blend"; os.makedirs(OUT, exist_ok=True)
def find(pat, must):
    c = [p for p in glob.glob(pat, recursive=True) if must in p]; assert c, (pat, must); print("нашёл:", c[0]); return c[0]
old = np.load(find("/kaggle/input/**/test_final_score.npy", "epic-blend-shark")).astype(np.float64)
nm = np.load(find("/kaggle/input/**/test_nmask.npy", "epic-v3-shark-pred")).astype(bool)
new = np.load(find("/kaggle/input/**/test_lograte.npy", "epic-v2-m3ft-shark")).astype(np.float64)
assert len(old) == len(new) == len(nm); ok = ~nm
print(f"позиций {len(nm):,}, N {int(nm.sum()):,}")
def z(x): mu, sd = x[ok].mean(), x[ok].std() + 1e-9; return (x - mu) / sd
zo, zn = z(old), z(new); del old, new
sub = np.random.default_rng(0).choice(np.flatnonzero(ok), 2_000_000, replace=False)
from scipy.stats import spearmanr
print("Spearman старый/новый на 2 млн позиций:", spearmanr(zo[sub], zn[sub])[0])
def rank_power(score, alpha):
    x = np.negative(score); x[nm] = np.inf
    order = np.argsort(x, kind="stable"); del x
    r = np.empty(len(order), np.float64); r[order] = np.arange(1, len(order) + 1, dtype=np.float64); del order
    np.power(r, -alpha, out=r); np.clip(r, 0, 1, out=r); r *= SCALE; np.rint(r, out=r)
    v = r.astype(np.int32); del r; v[nm] = 0; return v
def write(ints, path, chunk=8_000_000):
    with gzip.open(path, "wb", compresslevel=4) as fh:
        for a in range(0, len(ints), chunk):
            v = ints[a:a + chunk]; buf = np.empty((len(v), 8), np.uint8)
            buf[:, 0] = ord("0") + v // SCALE; buf[:, 1] = ord("."); buf[:, 7] = ord("\n")
            f = v % SCALE
            for k in range(6, 1, -1): buf[:, k] = ord("0") + f % 10; f //= 10
            fh.write(buf.tobytes())
for w, alpha in ((0.5, 0.65), (0.3, 0.65)):
    ints = rank_power(w * zn + (1 - w) * zo, alpha)
    name = f"{SPECIES}.predictions.blend_m3ft_w{w:.1f}_old.alpha{alpha:.2f}.txt.gz"
    path = f"{OUT}/{name}"; write(ints, path)
    with gzip.open(path, "rt") as fh: head = [next(fh).strip() for _ in range(3)]
    assert head == [f"{v / SCALE:.5f}" for v in ints[:3]]
    n = sum(1 for _ in gzip.open(path, "rt"))
    print(name, "строк", n, "= позиций", n == len(nm), "| [0,1]", bool(ints.min() >= 0 and ints.max() <= SCALE), "| N=0", bool((ints[nm] == 0).all()),
          "| уровней", len(np.unique(ints)), "| МБ", round(os.path.getsize(path) / 1e6, 1))
