"""RL learning curves from rsl_rl tensorboard logs: success against simulated hours.

    python -m analysis.learning_curves runs/rl/* --out results/learning_curves.png

Each run dir needs meta.json (written by sim/train_rl.py). Writes <run>/curve.csv per run and one
figure with a panel per task, a line per action space (thin: seeds, bold: mean over seeds).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

COLORS = {"raw": "#2a78d6", "vocab": "#eb6834"}     # fixed slots: colour follows the action space
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_run(run):
    meta = json.loads((run / "meta.json").read_text())
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    tags = [t for t in acc.Tags()["scalars"] if t.startswith(("Episode/", "Stage/"))]
    series = {t: {e.step: e.value for e in acc.Scalars(t)} for t in tags}
    steps = sorted(series.get("Episode/success", {}))
    rows = []
    for it in steps:
        row = {"iteration": it, "sim_hours": (it + 1) * meta["sim_seconds_per_iteration"] / 3600}
        row.update({t: series[t].get(it, float("nan")) for t in tags})
        rows.append(row)
    if rows:
        with open(run / "curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return meta, rows


def smooth(y, k=10):
    y = np.asarray(y, float)
    return np.convolve(np.nan_to_num(y), np.ones(k) / k, mode="valid") if len(y) >= k else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/learning_curves.png")
    args = ap.parse_args()

    groups = defaultdict(list)          # (task, action) -> list of (hours, success)
    for r in map(Path, args.runs):
        if not (r / "meta.json").exists():
            continue
        meta, rows = load_run(r)
        if rows:
            groups[(meta["task"], meta["action"])].append(
                ([row["sim_hours"] for row in rows], [row["Episode/success"] for row in rows]))
    tasks = sorted({t for t, _ in groups})
    if not tasks:
        raise SystemExit("no runs with logged success")

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for action, color in COLORS.items():
            runs = groups.get((task, action), [])
            if not runs:
                continue
            curves = []
            for hours, succ in runs:
                s = smooth(succ)
                h = np.asarray(hours[len(hours) - len(s):])
                ax.plot(h, s, color=color, lw=0.8, alpha=0.35)
                curves.append((h, s))
            n = min(len(s) for _, s in curves)
            mean = np.mean([s[:n] for _, s in curves], axis=0)
            h = curves[0][0][:n]
            ax.plot(h, mean, color=color, lw=2, label=f"{action} ({len(runs)} seed{'s' * (len(runs) > 1)})")
            ax.annotate(action, (h[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_title(task.upper(), fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("simulated hours", color=MUTED, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("success at episode end (training)", color=MUTED, fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
