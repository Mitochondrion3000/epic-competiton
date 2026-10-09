"""Финальная модель (шаг 6) в виде функций: для больших геномов, где обучение, предсказание теста и сборка сабмита
идут в разных процессах (например, в нескольких ноутбуках Kaggle параллельно). Логика та же, что в step6_final.ipynb."""
import gzip, hashlib, json, os, tarfile
import numpy as np, pandas as pd
import lightgbm as lgb
import features as F, scoring as S

CODE = np.full(256, 4, np.uint8)
for _i, _b in enumerate("ACGT"):
    CODE[ord(_b)] = CODE[ord(_b.lower())] = _i

LGB = dict(max_depth=6, num_leaves=64, learning_rate=0.05, min_data_in_leaf=200, bagging_fraction=0.8, bagging_freq=1,
           feature_fraction=0.8, verbose=-1, num_threads=os.cpu_count())
N_TREES = 800


def download_dataset(species, data_dir):
    if os.path.isdir(f"{data_dir}/{species}"): return
    import requests
    os.makedirs(data_dir, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == f"{species}.tgz"][0]
    dst, md5 = f"{data_dir}/{species}.tgz", hashlib.md5()
    with requests.get(f["links"]["self"], stream=True, timeout=120) as r, open(dst, "wb") as fh:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 22): fh.write(chunk); md5.update(chunk)
    assert md5.hexdigest() == f["checksum"].split(":")[-1], "MD5 не совпал"
    with tarfile.open(dst) as tf:
        try: tf.extractall(data_dir, filter="data")
        except TypeError: tf.extractall(data_dir)
    os.remove(dst)


def read_genome(path, keep):
    seq, low, name, chunks = {}, {}, None, []
    def flush():
        if name in keep:
            raw = np.frombuffer(b"".join(chunks), np.uint8)
            seq[name], low[name] = CODE[raw], raw >= ord("a")
    with gzip.open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                flush(); name, chunks = line[1:].split()[0].decode(), []
            elif name in keep:
                chunks.append(line.rstrip())
    flush()
    return seq, low


def read_template(path):
    t = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 5], names=["chrom", "start", "end", "strand"], dtype={"chrom": str, "strand": str})
    t["len"] = t.end - t.start; t["off"] = np.r_[0, np.cumsum(t.len.values)[:-1]]
    return t


def load_params(path):
    P = json.load(open(path))
    params = {"pwm": np.array(P["pwm"]), "kmers": [tuple(k) for k in P["kmers"]], "tf": P["tf"], "wide": P["wide"], "local": P["local"]}
    base = P["features"]
    return params, base, base + F.logo_names() + F.dinuc_names()


def extra(seq, pos, strand):
    return np.hstack([F.logo_onehot(seq, pos, strand), F.dinuc_onehot(seq, pos, strand)])


def build_rows(rows, SEQ, LOW, params, base_features, n_extra, log=print):
    """Признаки для строк таблицы: XB (float32, понятные признаки) и XE (uint8, логотип + пары букв)."""
    XB = np.zeros((len(rows), len(base_features)), np.float32)
    XE = np.zeros((len(rows), n_extra), np.uint8)
    for k, (c, d) in enumerate(rows.groupby("chrom", sort=False)):
        r = d.index.values
        XB[r] = F.feature_matrix(SEQ[c], LOW[c], d.pos.values, d.strand.values, params, base_features)
        for st in ("+", "-"):
            m = d.strand.values == st
            if m.any(): XE[r[m]] = extra(SEQ[c], d.pos.values[m], st)
        if k % 200 == 0: log(f"признаки: контиг {k}")
    return XB, XE


class Rows(lgb.Sequence):
    """«Ленивая» матрица для LightGBM: строки idx из XB и XE, склеенные по требованию (float64, как требует Sequence)."""
    def __init__(self, XB, XE, idx, batch_size=200_000):
        self.XB, self.XE, self.idx, self.batch_size = XB, XE, idx, batch_size
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        j = self.idx[k]
        if np.ndim(j) == 0: return np.concatenate([self.XB[j], self.XE[j]]).astype(np.float64)
        return np.hstack([self.XB[j], self.XE[j]]).astype(np.float64)


def train_pair(XB, XE, y, tot, idx, names, seed):
    """(шанс старта, сила старта среди стартов) на строках idx."""
    cls = lgb.train({**LGB, "objective": "binary", "seed": seed}, lgb.Dataset(Rows(XB, XE, idx), label=y[idx], feature_name=names), N_TREES)
    pidx = idx[y[idx] == 1]
    reg = lgb.train({**LGB, "objective": "regression", "seed": seed}, lgb.Dataset(Rows(XB, XE, pidx), label=np.log1p(tot[pidx]), feature_name=names), N_TREES)
    return cls, reg


def predict_positions(models, seq, low, pos, strand, params, base_features, chunk=1_000_000):
    """Счёт = среднее по моделям [log P(старт) + E[log(1 + счёт) | старт]] для позиций одной цепи одного контига."""
    base = F.feature_matrix(seq, low, pos, np.full(len(pos), strand), params, base_features)
    out = np.zeros(len(pos), np.float32)
    for a in range(0, len(pos), chunk):
        b = min(a + chunk, len(pos))
        Xc = np.hstack([base[a:b], extra(seq, pos[a:b], strand).astype(np.float32)])
        out[a:b] = np.mean([np.log(np.clip(c.predict(Xc), 1e-12, 1)) + r.predict(Xc) for c, r in models], axis=0)
    return out


def test_groups(tpl, n_parts):
    """Делим тестовые контиги на n_parts групп примерно равного размера (детерминированно)."""
    size = tpl.groupby("chrom", sort=False).len.sum().sort_values(ascending=False)
    load, part = np.zeros(n_parts), {}
    for c, n in size.items():
        k = int(np.argmin(load)); part[c] = k; load[k] += n
    return part


def predict_part(models, tpl, part_of, k, SEQ, LOW, params, base_features, log=print):
    """Счета для строк шаблона из группы k (в порядке шаблона среди этих строк); N-позиции = -inf."""
    sel = tpl[tpl.chrom.map(part_of) == k]
    out = np.zeros(int(sel.len.sum()), np.float32)
    local_off = np.r_[0, np.cumsum(sel.len.values)[:-1]]
    sel = sel.assign(loff=local_off)
    for j, ((c, st), g) in enumerate(sel.groupby(["chrom", "strand"], sort=False)):
        pos = np.concatenate([np.arange(s, e) for s, e in zip(g.start.values, g.end.values)])
        dst = np.concatenate([np.arange(o, o + n) for o, n in zip(g.loff.values, g.len.values)])
        sc = predict_positions(models, SEQ[c], LOW[c], pos, st, params, base_features)
        sc[SEQ[c][pos] == 4] = -np.inf
        out[dst] = sc
        if j % 50 == 0: log(f"часть {k}: {j} контиг×цепь")
    return out


def assemble(tpl, part_of, parts):
    """Склейка частей в полный вектор в порядке шаблона."""
    full = np.zeros(int(tpl.len.sum()), np.float32)
    for k, arr in parts.items():
        sel = tpl[tpl.chrom.map(part_of) == k]
        dst = np.concatenate([np.arange(o, o + n) for o, n in zip(sel.off.values, sel.len.values)])
        assert len(dst) == len(arr), f"часть {k}: длина не совпала"
        full[dst] = arr
    return full
