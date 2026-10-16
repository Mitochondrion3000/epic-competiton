# Место на лидерборде EPIC по виду: rank2 = порядок по ПРОИЗВЕДЕНИЮ мест по AUPRC и по Spearman среди всех сабмитов вида
# (проверено 2026-10-16 по /leaderboard/species/<id>: порядок команд совпадает с rank0_AUPRC * rank0_Spearman).
# Значит, вдвое лучшее место по любой метрике даёт одинаковый выигрыш: правило выбора alpha/смеси — минимум произведения.
# python tools/lb_rank.py                       -> по всем видам: наши места и сколько нужно прибавить по каждой метрике
# from tools.lb_rank import fetch, rank_product -> оценка кандидата (AUPRC, Spearman) против текущих сабмитов
import json, sys, urllib.request
import numpy as np

API = "https://epic.autosome.org/api/v1/leaderboard"
TEAM = "Mitochondrion"
SPECIES = ["Ofas_2.0", "ilPloInte3.2", "cgigas_uk_roslin_v1-Heart", "sSquSuc5.hap2", "ASM119413v2-Kidney"]

def fetch(species):
    # все сабмиты вида: [(команда, AUPRC, Spearman)]
    d = json.load(urllib.request.urlopen(f"{API}/species/{species}?limit=500&show=all", timeout=60))
    out = []
    for t in d["teams"]:
        s = {m["metric"]: m["score"] for m in t["metrics"]}
        if s.get("prauc") is not None and s.get("spearman") is not None: out.append((t["teamName"], s["prauc"], s["spearman"]))
    return out

def ranks(auprc, spear, subs, exclude_team=TEAM):
    # места кандидата среди чужих сабмитов (1 = лучший), их произведение и место команды по этому произведению
    other = [(a, s) for t, a, s in subs if t != exclude_team]
    A = np.array([a for a, _ in other]); S = np.array([s for _, s in other])
    ra, rs = 1 + int((A > auprc).sum()), 1 + int((S > spear).sum())
    best = {}                                                    # лучшее произведение каждой команды (их места тоже сдвигаются нами)
    for t, a, s in subs:
        if t == exclude_team: continue
        p = (1 + int((A > a).sum()) + (auprc > a)) * (1 + int((S > s).sum()) + (spear > s))
        best[t] = min(best.get(t, 1e18), p)
    return ra, rs, ra * rs, 1 + sum(v < ra * rs for v in best.values())

def rank_product(auprc, spear, species, subs=None):
    return ranks(auprc, spear, subs or fetch(species))

if __name__ == "__main__":
    for sp in SPECIES:
        subs = fetch(sp); ours = [(a, s) for t, a, s in subs if t == TEAM]
        if not ours: print(sp, "нет наших сабмитов"); continue
        a0, s0 = max(ours, key=lambda x: -np.prod(ranks(x[0], x[1], subs)[:2]))
        ra, rs, p, place = ranks(a0, s0, subs)
        line = f"{sp:28s} наш {a0:.3f}/{s0:.3f}: места AUPRC {ra}, Spearman {rs}, произведение {p}, место ~{place}"
        for da in (0.01, 0.02, 0.04):
            line += f" | +{da} AUPRC -> ~{ranks(a0 + da, s0, subs)[3]}"
        for ds in (0.01, 0.02, 0.04):
            line += f" | +{ds} Sp -> ~{ranks(a0, s0 + ds, subs)[3]}"
        print(line)
