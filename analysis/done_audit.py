"""13 Oct: long-horizon diagnosis tables from eval_chains --done_diag / --done_trace outputs.
    python -m analysis.done_audit runs/eval/longdiag_flag_unstack.csv [more csvs ...]
Per csv: done-flag precision by command position (from <csv>.diag.csv), time between consecutive marks, and from <csv>.trace.csv the flag
right after each mark (mean of the 10 ticks after it)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def load(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


for f in sys.argv[1:]:
    print(f"== {Path(f).name}")
    diag = [r for r in load(f + ".diag.csv") if int(r["ok"]) >= 0]
    if diag:
        by = defaultdict(list)
        for r in diag:
            by[int(r["command"])].append(int(r["ok"]))
        print("  marks really done, by command: " + "  ".join(f"#{k + 1} {sum(v)}/{len(v)} ({100 * sum(v) / len(v):.0f}%)" for k, v in sorted(by.items())))
        ticks = defaultdict(dict)
        for r in diag:
            ticks[int(r["env"])][int(r["command"])] = int(r["tick"])
        gaps = defaultdict(list)
        for t in ticks.values():
            for k in sorted(t):
                if k - 1 in t:
                    gaps[k].append(t[k] - t[k - 1])
        print("  ticks from mark k to mark k+1 (median, share <= 7 ticks): " + "  ".join(
            f"#{k}->#{k + 1} {median(v):.0f} ({100 * sum(g <= 7 for g in v) / len(v):.0f}%, n={len(v)})" for k, v in sorted(gaps.items())))
    tr = load(f + ".trace.csv")
    if tr:
        per = defaultdict(list)
        for r in tr:
            per[int(r["env"])].append((int(r["tick"]), int(r["marks"]), int(r["sim_cmd"]), float(r["done_p"])))
        after = defaultdict(list)
        for rows in per.values():
            rows.sort()
            for j in range(1, len(rows)):
                if rows[j][1] > rows[j - 1][1]:          # a mark was written at this tick
                    nxt = [p for (_, _, _, p) in rows[j + 1:j + 11]]
                    if nxt:
                        after[rows[j][1]].append(sum(nxt) / len(nxt))
        print("  flag in the 10 ticks after mark k (mean): " + "  ".join(f"after #{k} {sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(after.items())))
