"""Record expert rollouts with cameras for distillation into SmolVLA (B2, Ours, Oracle data).

Rolls out a policy on training layouts (never the eval states), keeps successful episodes
only, and writes one npz per episode in the replay format: front, wrist (T, H, W, 3) uint8,
state (T, 6), action (T, A), prompt (the combined-task prompt), skill sentence.
Raw-action policies record one frame per 0.1 s step (A = 5); vocabulary policies one frame
per primitive, with action = [skill one-hot | z] (A = n_skills + latent_dim).

    PYTHONPATH=. ./isaaclab.sh -p sim/record_rollouts.py --headless --enable_cameras --task c1 \
        --policy scripted --episodes 500 --out data/processed/rollouts/oracle_c1
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--policy", required=True, help="scripted | rsl:<checkpoint.pt>")
parser.add_argument("--vocab", default=None)
parser.add_argument("--episodes", type=int, default=500, help="successful episodes to keep")
parser.add_argument("--batch", type=int, default=100)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--image_size", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from sim.envs.vocab_env import PandaVocabEnv, PandaVocabEnvCfg  # noqa: E402
from sim.expert import SKILLS, ScriptedExpert  # noqa: E402


def main():
    n = args.batch
    if args.vocab:
        cfg = PandaVocabEnvCfg(task=args.task, cameras=True, image_size=args.image_size, vocab_path=args.vocab)
    else:
        cfg = PandaTaskEnvCfg(task=args.task, cameras=True, image_size=args.image_size)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaVocabEnv(cfg) if args.vocab else PandaTaskEnv(cfg)
    if args.policy == "scripted":
        policy = ScriptedExpert()
    else:
        from sim.rl_cfg import RslPolicy
        policy = RslPolicy(args.policy.removeprefix("rsl:"), env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    kept, tried = 0, 0
    while kept < args.episodes:
        env.set_layouts(tasks.sample_layouts(args.task, n, gen))
        obs, _ = env.reset()
        policy.reset(env)
        recs = [{"front": [], "wrist": [], "state": [], "action": [], "skills": []} for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool, device=env.device)
        for _ in range(int(env.max_episode_length) + 1):
            s = env.state()
            yaw = env.tcp_yaw()
            state = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08], -1)
            action, skill = policy.act(env)
            if args.vocab:              # record the executed primitive: one-hot of the argmax skill, clamped z
                k = len(env.skills)
                rec_action = torch.cat([torch.nn.functional.one_hot(action[:, :k].argmax(-1), k).float(),
                                        action[:, k:].clamp(-3, 3)], -1)
                names = [env.skills[i] for i in action[:, :k].argmax(-1).tolist()]
            else:
                rec_action = action.clamp(-1, 1)
                names = [SKILLS[i] for i in skill.tolist()] if skill is not None else [""] * n
            front, wrist = obs["front"][..., :3].to(torch.uint8).cpu(), obs["wrist"][..., :3].to(torch.uint8).cpu()
            for i in (~done).nonzero().squeeze(-1).tolist():
                r = recs[i]
                r["front"].append(front[i])
                r["wrist"].append(wrist[i])
                r["state"].append(state[i].cpu())
                r["action"].append(rec_action[i].cpu())
                if names[i] and (not r["skills"] or r["skills"][-1] != names[i]):
                    r["skills"].append(names[i])
            obs, *_ = env.step(action)
            policy.update(env)
            new = env.episode_done & ~done
            for i in new.nonzero().squeeze(-1).tolist():
                tried += 1
                if env.last_episode["success"][i] > 0 and kept < args.episodes:
                    r = recs[i]
                    np.savez_compressed(out / f"{args.task}_{kept:04d}.npz",
                                        **{k: torch.stack(r[k]).numpy() for k in ["front", "wrist", "state", "action"]},
                                        prompt=tasks.PROMPTS[args.task], skill=" > ".join(r["skills"]))
                    kept += 1
            done |= new
            if done.all():
                break
        print(f"kept {kept}/{args.episodes} successful episodes ({tried} tried)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
