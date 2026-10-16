import os, sys, gzip, tarfile, time, json, glob, io, urllib.request
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(os.cpu_count())
# 5.1 Архитектура
def code_table(device):
    # код 0..8 -> 6 каналов: one-hot ACGT (одинаковый для A и a), флаг строчной буквы, флаг N
    t = torch.zeros(9, 6, device=device)
    for b in range(4):
        t[b, b] = 1; t[b + 4, b] = 1; t[b + 4, 4] = 1
    t[8, 5] = 1
    return t

_TABLES = {}                          # таблица кодирования на каждом устройстве (модели ансамбля могут быть на разных GPU)
RC_PERM = [3, 2, 1, 0, 4, 5]          # A<->T, C<->G; флаги регистра и N от цепи не зависят

def encode(codes, use_lowercase=True):
    if codes.device not in _TABLES: _TABLES[codes.device] = code_table(codes.device)
    x = _TABLES[codes.device][codes.long()].permute(0, 2, 1)     # (B, 6, L)
    if not use_lowercase: x = x * torch.tensor([1, 1, 1, 1, 0, 1.], device=x.device).view(1, 6, 1)
    return x

def rc(x):
    return x[:, RC_PERM].flip(-1)                               # обратно-комплементарная последовательность

class ResBlock(nn.Module):
    # свёртка -> BN -> GELU -> свёртка -> BN, плюс вход (остаточная связь)
    def __init__(self, ch, k=5, d=1):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=d * (k // 2), dilation=d); self.n1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, k, padding=d * (k // 2), dilation=d); self.n2 = nn.BatchNorm1d(ch)
    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(x + self.n2(self.c2(h)))

class AttentionBottleneck(nn.Module):
    # self-attention по «токенам» самого грубого уровня; позиция кодируется depthwise-свёрткой (работает при любой длине окна)
    def __init__(self, ch, n_layers, n_heads):
        super().__init__()
        self.pos = nn.Conv1d(ch, ch, 3, padding=1, groups=ch)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(ch, n_heads, dim_feedforward=2 * ch, dropout=0.1,
                                                                activation="gelu", batch_first=True, norm_first=True)
                                     for _ in range(n_layers)])
    def forward(self, z):                                       # z: (B, C, T)
        h = (z + self.pos(z)).transpose(1, 2)
        for layer in self.layers: h = layer(h)
        return h.transpose(1, 2)

class InitiationNetV2(nn.Module):
    def __init__(self, ch_local=96, tower=(128, 192, 256, 320), stride=4, attn_layers=2, attn_heads=8, n_out=3, in_ch=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, ch_local, 11, padding=5), nn.BatchNorm1d(ch_local), nn.GELU())
        self.local = nn.Sequential(ResBlock(ch_local, 5, 1), ResBlock(ch_local, 5, 2))
        self.downs, self.blocks, self.ups = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev = ch_local
        for c in tower:
            self.downs.append(nn.Sequential(nn.Conv1d(prev, c, 2 * stride + 1, stride=stride, padding=stride), nn.BatchNorm1d(c), nn.GELU()))
            self.blocks.append(nn.Sequential(ResBlock(c, 5, 1), ResBlock(c, 5, 3)))
            self.ups.append(nn.Conv1d(c, ch_local, 1))
            prev = c
        self.attn = AttentionBottleneck(prev, attn_layers, attn_heads) if attn_layers > 0 else nn.Identity()
        self.head = nn.Sequential(nn.Conv1d(ch_local, ch_local, 1), nn.GELU(), nn.Conv1d(ch_local, n_out, 1))
        nn.init.constant_(self.head[-1].bias, -8.0)              # стартовая частота exp(-8) ~ 3e-4 на позицию, порядок реальной доли положительных

    def forward(self, x):                                       # x: (B, 6, L) -> (B, n_out, L) log-rate плюс-цепи
        L = x.shape[-1]
        h = self.local(self.stem(x)); fused, z = h, h
        for i, (down, blk, up) in enumerate(zip(self.downs, self.blocks, self.ups)):
            z = blk(down(z))
            if i == len(self.downs) - 1: z = self.attn(z)
            fused = fused + F.interpolate(up(z), size=L, mode="linear", align_corners=False)
        return self.head(fused)

def forward_both(net, x):
    # (B, K, 2, L): обе цепи одной сетью; минус-цепь = сеть на RC-окне, выход развёрнут обратно.
    # Окно и его RC идут одним батчем 2B: вдвое меньше запусков ядер GPU, чем два отдельных вызова
    B = x.shape[0]
    out = net(torch.cat([x, rc(x)]))
    return torch.stack([out[:B], out[B:].flip(-1)], dim=2)

def load_model_file(path, device):
    # число выходов берём из весов: модель могла учиться на виде с другим числом реплик или на двух видах
    ck = torch.load(path, map_location=device, weights_only=False); c = ck["cfg"]
    n_out = ck["model"]["head.2.weight"].shape[0]
    net = InitiationNetV2(c["CH_LOCAL"], tuple(c["TOWER"]), c["STRIDE"], c["ATTN_LAYERS"], c["ATTN_HEADS"], n_out)
    net.load_state_dict(ck["model"]); net.n_species = 2 if c.get("AUX_SPECIES") else 1; net.K = n_out // net.n_species
    return net.to(device).eval(), ck


SPECIES = os.environ.get("MOT_SPECIES", "ilPloInte3.2")
N_SITES = int(os.environ.get("MOT_N", "6000"))
CROP = int(os.environ.get("MOT_CROP", "200"))
RUN_MODISCO = os.environ.get("MOT_MODISCO", "1") == "1"
OUT = "/kaggle/working/motifs_" + SPECIES; os.makedirs(OUT, exist_ok=True)
ck_path = sorted(glob.glob("/kaggle/input/**/Ofas_2.0_v2moth_seed0.pt", recursive=True))[0]
net, ck = load_model_file(ck_path, "cpu"); cfg = ck["cfg"]
W = cfg["WIN"]; K = net.K
main_sp, aux_sp = "Ofas_2.0", cfg.get("AUX_SPECIES")
print("cfg WIN", W, "AUX", aux_sp, "n_species", net.n_species, "K", K, flush=True)
ch = 0 if SPECIES == main_sp else K
assert SPECIES == main_sp or SPECIES == aux_sp, "вид не обучен в этой модели"

# --- данные вида (Zenodo) ---
rec = json.load(urllib.request.urlopen("https://zenodo.org/api/records/22285753", timeout=60))
f = [x for x in rec["files"] if x["key"] == f"{SPECIES}.tgz"][0]
tgz = f"/tmp/{SPECIES}.tgz"
if not os.path.exists(tgz): urllib.request.urlretrieve(f["links"]["self"], tgz)
D = f"/tmp/d_{SPECIES}"; os.makedirs(D, exist_ok=True)
with tarfile.open(tgz) as tf:
    for m in tf:
        if m.isfile() and (m.name.endswith("genome.fa.gz") or "csRNA-" in m.name and m.name.endswith("-train.bed.gz")):
            tf.extract(m, D)
fa = glob.glob(f"{D}/**/genome.fa.gz", recursive=True)[0]
beds = sorted(glob.glob(f"{D}/**/csRNA-*r[0-9]-train.bed.gz", recursive=True)); print(beds, flush=True)
tabs = []
for b in beds:
    d = pd.read_csv(b, sep="\t", header=None, usecols=[0, 1, 2, 4, 5], names=["chrom", "start", "end", "count", "strand"], dtype={"chrom": str, "strand": str})
    span = (d.end - d.start).values; first = np.repeat(np.cumsum(span) - span, span)
    tabs.append(pd.DataFrame({"chrom": np.repeat(d.chrom.values, span), "pos": np.repeat(d.start.values, span) + (np.arange(span.sum()) - first),
                              "strand": np.repeat(d.strand.values, span), "count": np.repeat(d["count"].values, span).astype(np.float32)}))
lab = pd.concat(tabs).groupby(["chrom", "pos", "strand"], sort=False)["count"].sum().reset_index()
print("позиций с счётом:", len(lab), "медиана", lab["count"].median(), flush=True)
LUT = np.full(256, 8, np.uint8)
for i, bb in enumerate("ACGT"): LUT[ord(bb)] = i; LUT[ord(bb.lower())] = i + 4
need = set(lab.chrom); genome, name, chunks = {}, None, []
def flush():
    if name in need: genome[name] = LUT[np.frombuffer(b"".join(chunks), np.uint8)]
with gzip.open(fa, "rb") as fh:
    for line in fh:
        if line.startswith(b">"): flush(); name = line[1:].split()[0].decode(); chunks = []
        elif name in need: chunks.append(line.rstrip())
flush()
lab = lab[lab.chrom.isin(genome)]
lab = lab[lab.apply(lambda r: False, axis=1) == False] if False else lab
L = lab.chrom.map({c: len(v) for c, v in genome.items()}).values
lab = lab[(lab.pos.values > W) & (lab.pos.values < L - W)]
top = lab.sort_values("count", ascending=False).head(N_SITES * 3)
rng = np.random.default_rng(0); top = top.iloc[:N_SITES] if len(top) <= N_SITES else top.head(N_SITES)
print("порог счёта топа:", top["count"].min(), "сайтов", len(top), "минус-цепь", (top.strand == "-").mean(), flush=True)

c0 = W // 2
def window(chrom, pos, strand):
    g = genome[chrom]
    if strand == "+": s = pos - c0; return g[s:s + W]
    s = pos - (W - 1 - c0); w = g[s:s + W]
    return (3 - np.where(w < 4, w, np.where(w < 8, w - 4, 4)))[::-1] if False else w
tab = code_table("cpu")
def encode_win(w, strand):
    x = tab[torch.from_numpy(w.astype(np.int64))].T            # (6, W)
    return x if strand == "+" else x[RC_PERM].flip(-1)
t0 = time.time(); X, H = [], []
B = 16; rows = list(top.itertuples(index=False)); lo, hi = c0 - CROP, c0 + CROP
for i in range(0, len(rows), B):
    xs = []
    for r in rows[i:i + B]:
        s = r.pos - c0 if r.strand == "+" else r.pos - (W - 1 - c0)
        w = genome[r.chrom][s:s + W]
        if len(w) != W: continue
        xs.append(encode_win(w, r.strand))
    x = torch.stack(xs).requires_grad_(True)
    out = net(x)[:, ch, c0]; out.sum().backward()
    g = x.grad[:, :4, lo:hi]; oh = x.detach()[:, :4, lo:hi]
    hyp = g - g.mean(1, keepdim=True)
    H.append(hyp.permute(0, 2, 1).numpy().astype(np.float32)); X.append(oh.permute(0, 2, 1).numpy().astype(np.int8))
    if i % (B * 10) == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
H = np.concatenate(H); X = np.concatenate(X); A = H * X
np.savez_compressed(f"{OUT}/attr.npz", hyp=H, onehot=X)
off = np.arange(-CROP, CROP)
prof = np.abs(A).sum(2).mean(0); logo = A.mean(0)
np.save(f"{OUT}/pos_importance.npy", prof); np.save(f"{OUT}/mean_contrib_logo.npy", logo)
print("позиции с наибольшей важностью (смещение: значение):", [(int(off[j]), round(float(prof[j]), 4)) for j in np.argsort(-prof)[:15]], flush=True)
print(f"атрибуции готовы: {len(H)} сайтов за {time.time() - t0:.0f}s", flush=True)

if RUN_MODISCO:
    import modiscolite
    pos, neg = modiscolite.tfmodisco.TFMoDISco(hypothetical_contribs=H, one_hot=X.astype(np.float32), sliding_window_size=15, flank_size=5,
                                              target_seqlet_fdr=0.05, max_seqlets_per_metacluster=30000, n_leiden_runs=5, verbose=True)
    modiscolite.io.save_hdf5(f"{OUT}/modisco.h5", pos, neg, 15)
    ACG = np.array(list("ACGT"))
    res = []
    for sign, pats in (("pos", pos), ("neg", neg)):
        for k, p in enumerate(pats or []):
            ppm = np.array(p["sequence"]); cs = np.array(p["contrib_scores"])
            cons = "".join(ACG[ppm.argmax(1)]); ic = float((ppm * np.log2(np.clip(ppm, 1e-9, None) / 0.25)).sum())
            sl = p["seqlets"]; st = np.array(sl["start"]) + np.array(sl["end"]); cen = (st / 2 - CROP).mean() if len(sl["start"]) else np.nan
            res.append(dict(pattern=f"{sign}_{k}", seqlets=len(sl["start"]), consensus=cons, ic=round(ic, 1), mean_center_offset=round(float(cen), 1),
                            offset_sd=round(float((st / 2).std()), 1)))
    pd.DataFrame(res).to_csv(f"{OUT}/modisco_patterns.csv", index=False); print(pd.DataFrame(res).to_string(), flush=True)
