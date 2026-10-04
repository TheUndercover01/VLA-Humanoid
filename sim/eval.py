"""Run a policy on the saved eval start states and write one CSV row per episode.

All 100 start states run in parallel (one env each, one episode each).

    PYTHONPATH=. ./isaaclab.sh -p sim/eval.py --headless --task c1 --policy scripted
    ... --video media/c1_scripted.mp4 --enable_cameras      # also record envs 0 and 1
    ... --policy rsl:runs/rl/c1_raw_s1/model_1999.pt         # raw-action RL expert
    ... --policy rsl:runs/rl/c1_vocab_s1/model_249.pt --vocab data/processed/vocab_standin.pt
    ... --policy vla --enable_cameras [--plan]               # SmolVLA via vla/server.py (B0, B1, B1+LLM)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--policy", default="scripted", help="scripted | rsl:<checkpoint.pt> | vla")
parser.add_argument("--vocab", default=None, help="vocabulary for an rsl checkpoint trained with --action vocab")
parser.add_argument("--target_mode", action="store_true", help="vocabulary action with an end point (--action vocab_target)")
parser.add_argument("--vla_port", type=int, default=6011)
parser.add_argument("--plan", action="store_true", help="vla: follow the planner's atomic prompts (B1+LLM)")
parser.add_argument("--replan", type=int, default=3, help="vla: steps executed from each chunk before asking again")
parser.add_argument("--val", action="store_true", help="validation layouts (seed 4321) instead of the eval states, for tuning")
parser.add_argument("--name", default=None, help="model name in the CSV (default: --policy)")
parser.add_argument("--seed", type=int, default=0, help="training seed of the policy, for the CSV")
parser.add_argument("--out", default=None, help="CSV path (default runs/eval/<name>_<task>.csv)")
parser.add_argument("--video", default=None)
parser.add_argument("--video_envs", default="0,1")
parser.add_argument("--image_size", type=int, default=384)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from sim.envs.vocab_env import PandaVocabEnv, PandaVocabEnvCfg  # noqa: E402
from sim.expert import SKILLS, ScriptedExpert  # noqa: E402
from sim.video import Recorder  # noqa: E402

STATES = Path(__file__).parent / "eval_states"


def make_policy(name, env):
    if name == "scripted":
        return ScriptedExpert()
    if name.startswith("rsl:"):
        from sim.rl_cfg import RslPolicy
        return RslPolicy(name[4:], env)
    if name == "vla":
        from sim.vla_policy import VLAPolicy
        from vla.llm_planner import plan
        prompt = tasks.PROMPTS[args.task]
        # a vocabulary env step is a whole ~2 s primitive: ask for a new decision every step
        return VLAPolicy(args.vla_port, prompt, plan=plan(prompt) if args.plan else None,
                         replan=1 if args.vocab else args.replan)
    raise ValueError(f"unknown policy {name}")


def main():
    if args.val:
        layouts = tasks.sample_layouts(args.task, 100, torch.Generator().manual_seed(4321))
    else:
        layouts = torch.load(STATES / f"{args.task}.pt")["layouts"]
    n = len(layouts)
    vla = args.policy == "vla"
    cameras = args.video is not None or vla
    size = 256 if vla else args.image_size            # the VLA sees the image size it was trained on
    if args.vocab:
        cfg = PandaVocabEnvCfg(task=args.task, cameras=cameras, image_size=size, vocab_path=args.vocab,
                               target_mode=args.target_mode)
    else:
        cfg = PandaTaskEnvCfg(task=args.task, cameras=cameras, image_size=size)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaVocabEnv(cfg) if args.vocab else PandaTaskEnv(cfg)
    env.set_layouts(layouts)
    policy = make_policy(args.policy, env)
    obs, _ = env.reset()
    policy.reset(env)

    name = args.name or args.policy
    rec = None
    if args.video:
        ids = [int(i) for i in args.video_envs.split(",")]
        rec = Recorder(args.video, env_ids=ids, caption=f'{name}: "{tasks.PROMPTS[args.task]}"')
        env.tick_callback = lambda e: rec.add(e.images(), f"t = {e.ticks[ids[0]].item() * 0.1:4.1f} s")

    captured = torch.zeros(n, dtype=torch.bool, device=env.device)
    rows = [None] * n
    sentences = [[] for _ in range(n)]
    for t in range(int(env.max_episode_length) + 1):
        action, skill = policy.act(env)
        for i, k in enumerate(skill.tolist() if skill is not None else []):
            if not captured[i] and (not sentences[i] or sentences[i][-1] != SKILLS[k]):
                sentences[i].append(SKILLS[k])
        obs, _, _, _, _ = env.step(action)
        policy.update(env)
        if args.vocab:                                 # vocabulary policies: the chosen primitive
            for i, k in enumerate(env.skill.tolist()):
                if not captured[i] and (not sentences[i] or sentences[i][-1] != env.skills[k]):
                    sentences[i].append(env.skills[k])
        new = env.episode_done & ~captured
        if new.any():
            ids = new.nonzero().squeeze(-1)
            stages = tasks.failure_stage(args.task, env.last_reached[ids], env.last_knocked[ids])
            for j, i in enumerate(ids.tolist()):
                ok = bool(env.last_episode["success"][i])
                rows[i] = {
                    "model": name, "task": args.task, "seed": args.seed, "state": i, "success": int(ok),
                    "failure_stage": "" if ok else stages[j],
                    "time": round(env.last_episode["time"][i].item(), 2),
                    "jerk": round(env.last_episode["jerk"][i].item(), 3),
                    "peak_force": round(env.last_episode["peak_force"][i].item(), 2),
                    "lifted": int(env.last_episode["lifted"][i].item()),     # red cube was >= 5 cm up at some point
                    "skills": " > ".join(sentences[i]),
                }
            captured |= new
        if captured.all():
            break

    out = Path(args.out or f"runs/eval/{name}_{args.task}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    succ = sum(r["success"] for r in rows)
    fails = {}
    for r in rows:
        if not r["success"]:
            fails[r["failure_stage"]] = fails.get(r["failure_stage"], 0) + 1
    print(f"{name} on {args.task}: success {succ}/{n}, failures by stage {fails} -> {out}", flush=True)
    if rec:
        rec.save()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
