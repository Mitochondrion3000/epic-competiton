"""Единый eval-харнес EPIC (O-4): файл предсказаний + метки -> AUPRC/Spearman, alpha по правилу, калибровка tune->сайт, оценка мест.

Логика берётся из simple_model/scoring.py (официальные метрики) и tools/lb_rank.py (места по произведению мест, API сайта).
Скор -> rank_power(alpha) -> метрики -> места среди сабмитов вида.

Примеры:
  python tools/eval_harness.py --pred lograte.npy --gt gt.npy --species Ofas_2.0 --alphas 0.5 0.6 0.65 0.7 0.75 --n-ref 164000000
  python tools/eval_harness.py --selftest

Вход: --pred = .npy (лог-рейт/любой скор по позициям) или .txt/.txt.gz (число в строке);
      --gt = .npy (счёт csRNA-seq по позициям, >0 = старт); --mask = .npy bool (whitelist, по умолчанию все); --nmask = .npy bool (N-позиции -> 0).
Калибровка tune->сайт: CALIB = (0.877, 0.894) (множители AUPRC и Spearman, клоп, 2026-10-16); флаг --calib, применять к оценке на tune, не к report.
"""
import argparse, gzip, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from simple_model.scoring import SCALE, official_score  # noqa: E402

CALIB = (0.877, 0.894)


def rank_power(score, alpha, nmask=None, n_ref=None):
    """Как simple_model.scoring.rank_power; n_ref растягивает ранги до размера теста (метрики сопоставимы с лидербордом)."""
    x = score.astype(np.float32, copy=True)
    if nmask is not None: x[nmask] = -np.inf
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), np.float64); r[order] = np.arange(1, len(x) + 1); del order
    if n_ref: r = 1 + (r - 1) * (n_ref / len(x))
    s = (1.0 - (r - 1) / (n_ref or len(x))) if alpha == 0 else np.power(r, -alpha)
    s = np.rint(np.clip(s, 0, 1) * SCALE) / SCALE
    if nmask is not None: s[nmask] = 0.0
    return s.astype(np.float32)


def load_vec(path):
    if path.endswith(".npy"): return np.load(path)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh: return np.loadtxt(fh, dtype=np.float32)


def evaluate(score, gt, alphas, mask=None, nmask=None, n_ref=None, calib=None):
    """Список {alpha, AUPRC, Spearman} (с калибровкой tune->сайт, если задана calib=(kA, kS))."""
    rows = []
    for a in alphas:
        m = official_score(rank_power(score, a, nmask, n_ref), gt, mask)
        if calib: m = {"AUPRC": m["AUPRC"] * calib[0], "Spearman": m["Spearman"] * calib[1]}
        rows.append({"alpha": a, **m})
    return rows


def with_places(rows, species, subs=None):
    """Места среди чужих сабмитов вида (lb_rank.ranks): место AUPRC, место Spearman, произведение, место команды."""
    from lb_rank import fetch, ranks
    subs = subs or fetch(species)
    for r in rows:
        r["rank_auprc"], r["rank_spearman"], r["product"], r["team_place"] = ranks(r["AUPRC"], r["Spearman"], subs)
    return rows


def pick_alpha(rows):
    """Правило: минимум произведения мест; при равенстве выше AUPRC (после with_places)."""
    return min(rows, key=lambda r: (r["product"], -r["AUPRC"]))


def _selftest():
    rng = np.random.default_rng(0); n = 200_000
    gt = np.zeros(n); gt[rng.choice(n, 2000, replace=False)] = rng.integers(1, 5, 2000)
    score = (gt > 0) * 2.0 + rng.normal(size=n)
    rows = evaluate(score, gt, [0.3, 0.6, 0.9])
    assert all(0 < r["AUPRC"] <= 1 and -1 <= r["Spearman"] <= 1 for r in rows), rows
    assert evaluate(score, gt, [0.6], n_ref=n * 10)[0]["AUPRC"] > 0
    with_places(rows, None, subs=[("A", 0.20, 0.45), ("B", 0.10, 0.30), ("Mitochondrion", 0.15, 0.40)])
    print(pick_alpha(rows)); print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred"); ap.add_argument("--gt"); ap.add_argument("--mask"); ap.add_argument("--nmask")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8])
    ap.add_argument("--n-ref", type=int); ap.add_argument("--species"); ap.add_argument("--calib", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: _selftest(); sys.exit()
    score, gt = load_vec(a.pred), np.load(a.gt)
    mask = np.load(a.mask) if a.mask else None; nmask = np.load(a.nmask) if a.nmask else None
    rows = evaluate(score, gt, a.alphas, mask, nmask, a.n_ref, CALIB if a.calib else None)
    if a.species: rows = with_places(rows, a.species)
    for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    if a.species: print("alpha по правилу:", pick_alpha(rows)["alpha"])
