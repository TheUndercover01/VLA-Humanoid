"""Turn eval CSVs (sim/eval.py) into markdown tables: success with 95% Wilson CIs, failure stages, metrics.

    python -m analysis.results_table runs/eval/*.csv [--out results/table.md]

Episodes are pooled over training seeds; the seed count is shown per cell.
"""
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

TASK_ORDER = ["push", "lift", "c1", "c2", "c3"]
STAGE_ORDER = ["push", "reach", "grasp", "lift", "transport", "place", "knocked"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load(paths):
    cells = defaultdict(list)
    for p in paths:
        for row in csv.DictReader(open(p)):
            cells[(row["model"], row["task"])].append(row)
    return cells


def tables(cells):
    models = sorted({m for m, _ in cells})
    tasks = [t for t in TASK_ORDER if any(t == tk for _, tk in cells)]
    lines = ["## Success rate (95% Wilson CI)", "",
             "| Model | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for m in models:
        row = [m]
        for t in tasks:
            rows = cells.get((m, t), [])
            if not rows:
                row.append("–")
                continue
            k, n = sum(int(r["success"]) for r in rows), len(rows)
            lo, hi = wilson(k, n)
            seeds = len({r["seed"] for r in rows})
            row.append(f"{100 * k / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] ({seeds} seed{'s' * (seeds > 1)})")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Failure stage (share of all episodes)", "",
              "| Model | Task | " + " | ".join(STAGE_ORDER) + " |", "|---|---|" + "---|" * len(STAGE_ORDER)]
    for m in models:
        for t in tasks:
            rows = cells.get((m, t), [])
            if not rows:
                continue
            c = Counter(r["failure_stage"] for r in rows if r["success"] == "0")
            lines.append(f"| {m} | {t} | " + " | ".join(f"{100 * c[s] / len(rows):.0f}%" if c[s] else "" for s in STAGE_ORDER)
                         + " |")

    lines += ["", "## Secondary metrics (successful episodes, mean)", "",
              "| Model | Task | time (s) | EE jerk (m/s³) | peak contact force (N) |", "|---|---|---|---|---|"]
    for m in models:
        for t in tasks:
            ok = [r for r in cells.get((m, t), []) if r["success"] == "1"]
            if ok:
                mean = lambda k: sum(float(r[k]) for r in ok) / len(ok)  # noqa: E731
                lines.append(f"| {m} | {t} | {mean('time'):.1f} | {mean('jerk'):.2f} | {mean('peak_force'):.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    md = tables(load(args.csvs))
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()
