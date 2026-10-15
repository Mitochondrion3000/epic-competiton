# Сборка HTML-отчёта: python tools/build_error_report.py error_stats_v2.json error_stats_nem.json out.html
import json, re, sys
v2, nem, out = sys.argv[1:4]
R = "https://github.com/Mitochondrion3000/epic-competiton/blob/main/tools/"
src = open("tools/error_stats.py", encoding="utf-8").read().split("\n")
mark = lambda s: next(i + 1 for i, l in enumerate(src) if s in l)
cuts = [("tol", "# --- допуск"), ("str", "# --- доля пропусков"), ("cl", "# --- одиночные"), ("pr", "# --- PR"), ("pred", "# --- распределение"), ("feat", "# --- группы для признаков"), ("nuc", "    R[\"nuc\"] =")]
starts = {k: mark(m) for k, m in cuts}; order = [k for k, _ in cuts]; ends = {k: (starts[order[i + 1]] - 1 if i + 1 < len(order) else len(src)) for i, k in enumerate(order)}
links = {k: dict(url=f"{R}error_stats.py#L{starts[k]}-L{ends[k]}", label=f"tools/error_stats.py:{starts[k]}-{ends[k]}") for k in order}
links["feat"]["url"] = f"{R}error_stats.py#L{starts['feat']}-L{starts['nuc']}"; links["feat"]["label"] = f"tools/error_stats.py:{starts['feat']}-{starts['nuc']}"
links["defs"] = dict(url=f"{R}error_stats.py#L{mark('thr = np.partition')-1}-L{mark('# --- допуск')-1}", label="tools/error_stats.py (определения групп)")
html = open("tools/report/template.html", encoding="utf-8").read()
html = html.replace("/*DATA*/null", json.dumps(dict(v2=json.load(open(v2)), nem=json.load(open(nem))), ensure_ascii=False)).replace("/*LINKS*/null", json.dumps(links, ensure_ascii=False))
open(out, "w", encoding="utf-8").write(html); print("ok", len(html))
