"""Run a skill policy on fixed skill chains, most of them longer than anything it was trained on.

The skill env switches to the next command once the current one is done (the clips' end state,
privileged sim state, like the B1+LLM completion check). Success is the chain's last command done
and, for c1/c2/c3/unstack, also the task's own tasks.success check. Chains (fixed before any result):

  lift     reach r > pick-lift r                                   (2: training length, sanity check)
  c1       reach r > pick-lift r > place-down r on target          (3)
  c2       reach r > pick-lift r > place-down r on blue            (3)
  c3       push b to target > reach r > pick-lift r > place-down r on blue    (4)
  swap     reach b > pick-lift b > place-down b on red             (3: blue on red, roles reversed)
  unstack  c2, then reach r > pick-lift r > place-down r on target (6)
  restack  unstack, then reach b > pick-lift b > place-down b on red   (9)

    PYTHONPATH=. ./isaaclab.sh -p sim/eval_chains.py --headless --chain c1 \
        --policy rsl:runs/rl/skills_clip_chain2_s1/model_2999.pt [--val]
Writes runs/eval/<name>_chain_<chain>.csv: one row per start state.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--chain", required=True)
parser.add_argument("--policy", required=True, help="rsl:<checkpoint.pt>")
parser.add_argument("--val", action="store_true", help="validation layouts (seed 4321) instead of the eval states")
parser.add_argument("--name", default=None)
parser.add_argument("--out", default=None)
parser.add_argument("--ref", default="data/processed/skill_ref_standin.npz")
parser.add_argument("--bank", default="data/processed/state_bank_standin.npz")
parser.add_argument("--video", default=None)
parser.add_argument("--video_envs", default="0,1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from sim.clips import SKILLS  # noqa: E402
from sim.envs import tasks  # noqa: E402
from sim.envs.skill_env import NONE, OTHER, TARGET, PandaSkillEnv, PandaSkillEnvCfg  # noqa: E402
from sim.rl_cfg import RslPolicy  # noqa: E402

R, B = 0, 1
STACK = [("reach", R, NONE), ("pick-lift", R, NONE), ("place-down", R, OTHER)]
TO_TARGET = [("reach", R, NONE), ("pick-lift", R, NONE), ("place-down", R, TARGET)]
BLUE_ON_RED = [("reach", B, NONE), ("pick-lift", B, NONE), ("place-down", B, OTHER)]
# name: (commands, layouts from, final tasks.success check or "")
CHAINS = {
    "lift": (TO_TARGET[:2], "c1", "lift"),
    "c1": (TO_TARGET, "c1", "c1"),
    "c2": (STACK, "c2", "c2"),
    "c3": ([("push", B, TARGET)] + STACK, "c3", "c3"),
    "swap": (BLUE_ON_RED, "c2", ""),
    "unstack": (STACK + TO_TARGET, "c2", "c1"),
    "restack": (STACK + TO_TARGET + BLUE_ON_RED, "c2", ""),
}
SECONDS_PER_COMMAND = 6.0
STATES = Path(__file__).parent / "eval_states"


def main():
    chain, layout_task, check = CHAINS[args.chain]
    if args.val:
        layouts = tasks.sample_layouts(layout_task, 100, torch.Generator().manual_seed(4321))
    else:
        layouts = torch.load(STATES / f"{layout_task}.pt")["layouts"]
    n = len(layouts)
    cfg = PandaSkillEnvCfg(task=layout_task, ref_path=args.ref, bank_path=args.bank, final_check=check,
                           episode_s=SECONDS_PER_COMMAND * len(chain) + 5, cameras=args.video is not None,
                           image_size=384)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaSkillEnv(cfg)
    env.set_chain(chain)
    env.set_layouts(layouts)
    policy = RslPolicy(args.policy[4:], env)
    env.reset()
    name = args.name or Path(args.policy[4:]).parent.name

    rec = None
    if args.video:
        from sim.video import Recorder
        ids = [int(i) for i in args.video_envs.split(",")]
        text = " > ".join(f"{s} {'rb'[c]}" for s, c, _ in chain)
        rec = Recorder(args.video, env_ids=ids, caption=f"{name}: {args.chain} ({text})")
        env.tick_callback = lambda e: rec.add(
            e.images(), f"t = {e.ticks[ids[0]].item() * 0.1:4.1f} s  command {e.cur[ids[0]].item() + 1}/{len(chain)}: "
                        f"{SKILLS[e.command()[0][ids[0]].item()]}")

    captured = torch.zeros(n, dtype=torch.bool, device=env.device)
    rows = [None] * n
    for _ in range(int(env.max_episode_length) + 1):
        action, _ = policy.act(env)
        env.step(action)
        new = env.episode_done & ~captured
        for i in new.nonzero().squeeze(-1).tolist():
            ok = bool(env.last_episode["success"][i])
            done = int(env.last_episode["completed"][i])
            rows[i] = {"model": name, "chain": args.chain, "state": i, "success": int(ok), "commands_done": done,
                       "n_commands": len(chain),
                       "failed_at": "" if ok else (f"{done + 1}:{chain[done][0]}" if done < len(chain) else "final check"),
                       "knocked": int(env.last_knocked[i]),
                       "time": round(env.last_episode["time"][i].item(), 2),
                       "jerk": round(env.last_episode["jerk"][i].item(), 3),
                       "peak_force": round(env.last_episode["peak_force"][i].item(), 2)}
        captured |= new
        if captured.all():
            break

    out = Path(args.out or f"runs/eval/{name}_chain_{args.chain}{'_val' if args.val else ''}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    succ = sum(r["success"] for r in rows)
    fails = {}
    for r in rows:
        if not r["success"]:
            fails[r["failed_at"]] = fails.get(r["failed_at"], 0) + 1
    mean_done = sum(r["commands_done"] for r in rows) / n
    print(f"{name} on {args.chain} ({len(chain)} commands): success {succ}/{n}, mean commands done {mean_done:.2f}, "
          f"failures at {dict(sorted(fails.items()))} -> {out}", flush=True)
    if rec:
        rec.save()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
