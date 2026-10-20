import glob, gzip, os, numpy as np
SCALE = 100_000
OUT = "/kaggle/working/moth_blend_v3"; os.makedirs(OUT, exist_ok=True)
def find(pat, must):
    c = [p for p in glob.glob(pat, recursive=True) if must in p]; assert c, (pat, must); print("нашёл:", c[0]); return c[0]
old = np.load(find("/kaggle/input/**/test_final_score.npy", "epic-v3-moth-blend")).astype(np.float64)
nm = np.load(find("/kaggle/input/**/test_nmask.npy", "epic-v3-moth-blend")).astype(bool)
new = np.load(find("/kaggle/input/**/test_lograte.npy", "epic-v2-m3ft-moth")).astype(np.float64)
assert len(old) == len(new) == len(nm) == len(v3); ok = ~nm
v3 = np.load(find("/kaggle/input/**/test_final_score.npy", "epic-v3-m6-pred-moth")).astype(np.float64)
print(f"позиций {len(nm):,}, N {int(nm.sum()):,}")
def z(x): mu, sd = x[ok].mean(), x[ok].std() + 1e-9; return (x - mu) / sd
zo, zn, z3 = z(old), z(new), z(v3); del old, new, v3
best = z(0.5 * zn + 0.5 * zo); del zo, zn   # лучший на лидерборде: вес m3ft 0.5, alpha 0.70
sub = np.random.default_rng(0).choice(np.flatnonzero(ok), 2_000_000, replace=False)
from scipy.stats import spearmanr
print("Spearman лучший/v3 на 2 млн позиций:", spearmanr(best[sub], z3[sub])[0])
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
for w, alpha in ((0.3, 0.70), (0.5, 0.70)):
    ints = rank_power(w * z3 + (1 - w) * best, alpha)
    name = f"ilPloInte3.2.predictions.blend_v3_w{w:.1f}_best.alpha{alpha:.2f}.txt.gz"
    path = f"{OUT}/{name}"; write(ints, path)
    with gzip.open(path, "rt") as fh: head = [next(fh).strip() for _ in range(3)]
    assert head == [f"{v / SCALE:.5f}" for v in ints[:3]]
    n = sum(1 for _ in gzip.open(path, "rt"))
    print(name, "строк", n, "= позиций", n == len(nm), "| [0,1]", bool(ints.min() >= 0 and ints.max() <= SCALE), "| N=0", bool((ints[nm] == 0).all()),
          "| уровней", len(np.unique(ints)), "| МБ", round(os.path.getsize(path) / 1e6, 1))
