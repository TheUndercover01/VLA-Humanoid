"""One-screen status of skill-chain RL runs: iteration, speed, ETA, reward and per-skill completion.

    python -m analysis.rl_status                       # every runs/rl/skills_* run
    python -m analysis.rl_status skills_clip_chain2_s1
"""
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SKILLS = ["reach", "push", "pick-lift", "place-down"]


def status(run):
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    if "Train/mean_reward" not in tags:
        return f"{run.name}: no iterations logged yet"
    rew = acc.Scalars("Train/mean_reward")
    n = len(rew)
    k = min(20, n - 1)
    s_per_it = (rew[-1].wall_time - rew[-1 - k].wall_time) / k if k else float("nan")
    total = json.loads((run / "meta.json").read_text()).get("max_iterations", 3000)
    total = 3000 if total is None else total
    lines = [f"{run.name}: iteration {n}, {s_per_it:.1f} s/iter, ~{(total - n) * s_per_it / 3600:.1f} h to {total}"]
    marks = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    lines.append("  iteration      " + " ".join(f"{rew[i].step:>6d}" for i in marks))
    lines.append("  mean reward    " + " ".join(f"{rew[i].value:6.1f}" for i in marks))
    for s in SKILLS:
        tag = f"Skill/{s}"
        if tag in tags:
            v = acc.Scalars(tag)
            # per-skill rates are logged once per iteration with episode ends; average the last 10 for noise
            last = sum(x.value for x in v[-10:]) / len(v[-10:])
            lines.append(f"  {s:12s} done " + " ".join(f"{v[min(i, len(v) - 1)].value:6.0%}" for i in marks)
                         + f"   (last 10 avg {last:.0%})")
    if "Episode/success" in tags:
        v = acc.Scalars("Episode/success")
        lines.append("  chain done     " + " ".join(f"{v[min(i, len(v) - 1)].value:6.0%}" for i in marks))
    return "\n".join(lines)


def main():
    root = Path("runs/rl")
    runs = [root / a for a in sys.argv[1:]] or sorted(root.glob("skills_*"))
    print("\n".join(status(r) for r in runs))


if __name__ == "__main__":
    main()
