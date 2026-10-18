# ---- драйвер: акула sSquSuc5.hap2, группа report (30 Мбнт выборки HO_MAX_BP, как в ft2), AlphaGenome vs v2 ft2 ----
SP = "sSquSuc5.hap2"; D = "/tmp/epic"; B = f"{D}/{SP}"; FP = f"{B}/for_participants"; OUT = "/kaggle/working"
BUDGET = int(os.environ.get("AG_BUDGET", "60")); HO_MAX_BP = 30_000_000
FAKE = os.environ.get("AG_FAKE") == "1"          # проверка кода без API: случайные «предсказания»

def make_rows(df):
    r = df[["chrom", "start", "end"]].copy()
    r["strand"] = df["strand"].values.astype(int)
    r["len"] = r.end - r.start; r["off"] = np.r_[0, np.cumsum(r.len.values)[:-1]]
    return r.reset_index(drop=True)

def report_rows(wl, contigs):
    # ровно как в epic_v2_model.ipynb: train_rows(GROUPS["report"]) + случайная выборка интервалов (random_state=0)
    iv = wl[wl.chrom.isin(set(contigs))]
    rows = make_rows(pd.concat([iv.assign(strand=0), iv.assign(strand=1)]))
    if rows.len.sum() / 2 > HO_MAX_BP:
        plus = rows[rows.strand == 0].sample(frac=1, random_state=0)
        keep = plus[np.cumsum(plus.len.values) <= HO_MAX_BP]
        sub = pd.concat([keep, keep.assign(strand=1)]).sort_values(["strand", "chrom", "start"])
        rows = make_rows(sub.drop(columns=["len", "off"]))
    return rows

def gather_rows(rows, contig_fn, dtype=np.float32):
    out = np.zeros(int(rows.len.sum()), dtype)
    for c, g in rows.groupby("chrom", sort=False):
        lo, hi = int(g.start.min()), int(g.end.max()); arr = contig_fn(c, lo, hi)
        for s, e, st, o in zip(g.start.values, g.end.values, g.strand.values, g.off.values):
            out[o:o + e - s] = arr[st, s - lo:e - lo]
    return out

def row_windows(rows):
    # окна 1 Мнт, пересекающие строки (плюс-цепь); вес = число позиций выборки в окне
    cnt = {}
    for c, s0, e0 in zip(rows.chrom.values[rows.strand.values == 0], rows.start.values[rows.strand.values == 0], rows.end.values[rows.strand.values == 0]):
        for w0 in range(s0 // W * W, e0, W):
            cnt[(c, w0)] = cnt.get((c, w0), 0) + min(e0, w0 + W) - max(s0, w0)
    return cnt

def scores_on_rows(rows, cache, key):
    sc = np.full(int(rows.len.sum()), np.nan, np.float32)
    for c, s0, e0, st, o in zip(rows.chrom.values, rows.start.values, rows.end.values, rows.strand.values, rows.off.values):
        for w0 in range(s0 // W * W, e0, W):
            d = cache.get((c, w0))
            if d is None or (key[0], key[1], st) not in d: continue
            a, b = max(s0, w0), min(e0, w0 + W)
            sc[o + a - s0:o + b - s0] = d[(key[0], key[1], st)][a - w0:b - w0]
    return sc

def analyse(rows, wins, cache, p2, gt, nm, out_dir):
    """AG отдельно и смесь z_ft2 + w * z_ag; половины = чётные/нечётные окна; w выбирается на одной половине, оценка на другой."""
    widx = {w: i for i, w in enumerate(wins)}
    win_of = np.full(len(gt), -1, np.int32)
    for c, s0, e0, st, o in zip(rows.chrom.values, rows.start.values, rows.end.values, rows.strand.values, rows.off.values):
        w0 = s0 // W * W
        if (c, w0) in widx and e0 <= w0 + W: win_of[o:o + e0 - s0] = widx[(c, w0)]       # строка целиком в окне (прочие делим по началу)
        elif (c, w0) in widx: win_of[o:o + e0 - s0] = widx[(c, w0)]
    tab = []
    for t in TYPES:
        for how in ("mean", "max"):
            s = scores_on_rows(rows, cache, (t, how)); cov = np.isfinite(s) & ~nm
            tab.append(dict(type=t, agg=how, n=int(cov.sum()), pos=int((gt[cov] > 0).sum()), rand=float((gt[cov] > 0).mean()),
                            AG_AUPRC_raw=float(average_precision_score(gt[cov] > 0, s[cov])), **{f"AG_{k}": v for k, v in official_score(rank_power(s[cov], 0.65), gt[cov]).items()},
                            ft2_AUPRC_raw=float(average_precision_score(gt[cov] > 0, p2[cov])), **{f"ft2_{k}": v for k, v in official_score(rank_power(p2[cov], 0.65), gt[cov]).items()}))
    tab = pd.DataFrame(tab); tab.to_csv(f"{out_dir}/ag_shark_metrics.csv", index=False); print(tab.round(4).to_string(), flush=True)
    S = {t: scores_on_rows(rows, cache, (t, "mean")) for t in TYPES}
    cov = np.isfinite(S["CAGE"]) & np.isfinite(S["PROCAP"]) & ~nm & (win_of >= 0)
    zs = lambda v: (v - v[cov].mean()) / v[cov].std()
    z2 = zs(p2); res = []
    for t in TYPES:
        za = zs(np.log1p(np.maximum(S[t], 0)))
        for w in (0.0, 0.1, 0.2, 0.35, 0.5, 1.0):
            sc = z2 + w * za
            for a in (0.5, 0.65):
                for hn, hm in (("even", win_of % 2 == 0), ("odd", win_of % 2 == 1), ("all", np.ones(len(gt), bool))):
                    ii = np.flatnonzero(cov & hm)
                    if len(ii): res.append(dict(type=t, w=w, alpha=a, half=hn, **official_score(rank_power(sc[ii], a), gt[ii])))
    R = pd.DataFrame(res); R.to_csv(f"{out_dir}/ag_shark_combine.csv", index=False); print(R[R.half == "all"].round(4).to_string(), flush=True)
    out = []
    for t in TYPES:
        for ev, tr in (("even", "odd"), ("odd", "even")):
            g = R[(R.type == t) & (R.half == tr)]; best = g.sort_values("AUPRC", ascending=False).iloc[0]
            e = R[(R.type == t) & (R.half == ev)]
            base = e[(e.w == 0.0) & (e.alpha == best.alpha)].iloc[0]; pick = e[(e.w == best.w) & (e.alpha == best.alpha)].iloc[0]
            out.append(dict(type=t, eval=ev, w=best.w, alpha=best.alpha, AUPRC_ft2=base.AUPRC, AUPRC_comb=pick.AUPRC, Sp_ft2=base.Spearman, Sp_comb=pick.Spearman, dAUPRC_pct=(pick.AUPRC / base.AUPRC - 1) * 100))
    O = pd.DataFrame(out); print(O.round(4).to_string(), flush=True); O.to_csv(f"{out_dir}/ag_shark_halves.csv", index=False)
    return tab, R, O

def main():
    gj = glob.glob("/kaggle/input/**/holdout_groups.json", recursive=True)[0]; npy = glob.glob("/kaggle/input/**/holdout_report_lograte.npy", recursive=True)[0]
    contigs = json.load(open(gj))["report"]; p2 = np.load(npy).astype(np.float32).ravel(); print("report-контиги:", contigs, "| предсказание ft2:", p2.shape, flush=True)
    client = None
    if not FAKE:
        os.system("pip install -q alphagenome")
        client = make_client(); print("ключ найден, клиент создан", flush=True)         # раньше загрузки данных: без секрета падаем сразу
    if not os.path.isdir(B):
        os.makedirs(D, exist_ok=True)
        rec = requests.get("https://zenodo.org/api/records/22285753", timeout=60).json()
        f = [x for x in rec["files"] if x["key"] == f"{SP}.tgz"][0]; dst = f"{D}/{SP}.tgz"
        for attempt in range(10):
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
            if have >= f["size"]: break
            try:
                with requests.get(f["links"]["self"], stream=True, timeout=120, headers={"Range": f"bytes={have}-"} if have else {}) as r, open(dst, "ab" if have else "wb") as fh:
                    r.raise_for_status()
                    if have and r.status_code != 206: fh.truncate(0)
                    for ch in r.iter_content(1 << 22): fh.write(ch)
            except requests.exceptions.RequestException as e:
                print("обрыв загрузки", e.__class__.__name__, flush=True)
        with tarfile.open(dst) as tf: tf.extractall(D)
        os.remove(dst)
    wl = pd.read_csv(f"{FP}/whitelist.train.bed.gz", sep="\t", header=None, usecols=range(3), names=["chrom", "start", "end"], dtype={"chrom": str})
    rows = report_rows(wl, contigs); print("строк:", len(rows), "позиций:", int(rows.len.sum()), flush=True)
    assert int(rows.len.sum()) == len(p2), f"число позиций не совпало с ft2: {int(rows.len.sum())} против {len(p2)}"
    LUT = np.full(256, 8, np.uint8)
    for i, b in enumerate("ACGT"): LUT[ord(b)] = i; LUT[ord(b.lower())] = i + 4
    genome, name, ch, keep = {}, None, [], set(contigs)
    def flush():
        if name in keep: genome[name] = LUT[np.frombuffer(b"".join(ch), np.uint8)]
    with gzip.open(f"{B}/genome/genome.fa.gz", "rb") as fh:
        for l in fh:
            if l.startswith(b">"): flush(); name = l[1:].split()[0].decode(); ch = []
            elif name in keep: ch.append(l.rstrip())
    flush(); assert all(c in genome for c in contigs)
    # метки: сумма реплик по позициям-цепям report-контигов
    tot = {}
    for path in sorted(glob.glob(f"{FP}/csRNA-*r[0-9]-train.bed.gz")):
        d = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 4, 5], names=["chrom", "start", "end", "count", "strand"], dtype={"chrom": str, "strand": str})
        d = d[d.chrom.isin(keep)]; span = (d.end - d.start).values; first = np.repeat(np.cumsum(span) - span, span)
        pos = np.repeat(d.start.values, span) + (np.arange(span.sum()) - first)
        x = pd.DataFrame({"chrom": np.repeat(d.chrom.values, span), "pos": pos, "st": np.repeat((d.strand.values == "-").astype(np.int8), span), "c": np.repeat(d["count"].values, span).astype(np.float32)})
        for (c, st), g in x.groupby(["chrom", "st"]):
            k = tot.setdefault((c, int(st)), {})
            for p_, v_ in zip(g.pos.values, g.c.values): k[p_] = k.get(p_, 0.0) + v_
    def tg(c, lo, hi):
        a = np.zeros((2, hi - lo), np.float32)
        for st in (0, 1):
            k = tot.get((c, st))
            if k:
                pp = np.fromiter(k.keys(), np.int64, len(k)); vv = np.fromiter(k.values(), np.float32, len(k)); m = (pp >= lo) & (pp < hi); a[st, pp[m] - lo] = vv[m]
        return a
    gt = gather_rows(rows, tg); nm = gather_rows(rows, lambda c, lo, hi: np.broadcast_to(genome[c][lo:hi] == 8, (2, hi - lo)), bool)
    print("позиций:", len(gt), "положительных:", int((gt > 0).sum()), "N:", int(nm.sum()), flush=True)
    chk = float(average_precision_score(gt[~nm] > 0, p2[~nm])); print(f"КОНТРОЛЬ: AUPRC(сырое) ft2 по восстановленным меткам {chk:.5f}; в отчёте ft2 0.10223", flush=True)
    assert abs(chk - 0.10223) < 2e-4, "метки или строки не совпали с ft2"
    cnt = row_windows(rows); wins = [w for w, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:BUDGET]]
    print(f"окон всего {len(cnt)}, берём {len(wins)}, покрытие выборки {sum(cnt[w] for w in wins) / sum(cnt.values()):.1%}", flush=True)
    cache = {}
    if FAKE:
        rng = np.random.default_rng(0)
        pred = lambda cl, seq, types: {t: (rng.random((W, 2), dtype=np.float32), np.array(["+", "-"])) for t in types}
    else: pred = ag_predict
    for i in range(0, len(wins), 5):
        predict_windows(client, genome, wins[i:i + 5], cache, predict=pred)
        np.savez_compressed(f"{OUT}/ag_shark_cache.npz", **{f"{c}|{s}|{'|'.join(map(str, k))}": v.astype(np.float16) for (c, s), d in cache.items() for k, v in d.items()})
        print("готово окон", len(cache), flush=True)
    analyse(rows, wins, cache, p2, gt, nm, OUT)

if __name__ == "__main__": main()
