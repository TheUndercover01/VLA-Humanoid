"""PPO (rsl_rl) on the raw 5-D action: the B2 teachers and the "RL experts" reference row.

    PYTHONPATH=. ./isaaclab.sh -p sim/train_rl.py --headless --task c1 --seed 1 --max_iterations 1500

Logs (tensorboard) and checkpoints go to runs/rl/<task>_raw_s<seed>/. One iteration is
num_steps_per_env * num_envs control steps of 0.1 s each, which gives the simulated-seconds axis.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--run_name", default=None)
parser.add_argument("--action", default="raw", choices=["raw", "vocab", "vocab_target"])
parser.add_argument("--vocab", default="data/processed/vocab_standin.pt")
parser.add_argument("--resume", default=None, help="checkpoint to continue from (same run dir)")
parser.add_argument("--skill_reward", default="clip", choices=["clip", "sparse"],
                    help="--task skills: reward from the clips, or only the clip-defined done events (ablation)")
parser.add_argument("--max_chain", type=int, default=2, help="--task skills: commands per training episode")
parser.add_argument("--hold_fix", action="store_true", help="--task keypoints: reward for holding the cube at the goal, no speed check")
parser.add_argument("--ref", default=None, help="skill/keypoint reference (default: the stand-in one for --task)")
parser.add_argument("--bank", default="data/processed/state_bank_standin.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.ref is None:
    args.ref = "data/processed/keypoint_ref_standin.npz" if args.task == "keypoints" else "data/processed/skill_ref_standin.npz"
app = AppLauncher(args).app

import json  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.envs.skill_env import PandaSkillEnv, PandaSkillEnvCfg  # noqa: E402
from sim.envs.vocab_env import PandaVocabEnv, PandaVocabEnvCfg  # noqa: E402
from sim.rl_cfg import RawPPOCfg, VocabPPOCfg  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    skills = args.task in ("skills", "keypoints")
    if args.task == "keypoints":
        env_cfg = PandaKeypointEnvCfg(task="c1", terminate_on_success=False, max_chain=args.max_chain,
                                      ref_path=args.ref, bank_path=args.bank, hold_fix=args.hold_fix)
    elif skills:
        env_cfg = PandaSkillEnvCfg(task="c1", terminate_on_success=False, reward=args.skill_reward,
                                   max_chain=args.max_chain, ref_path=args.ref, bank_path=args.bank)
    elif args.action.startswith("vocab"):
        env_cfg = PandaVocabEnvCfg(task=args.task, terminate_on_success=False, vocab_path=args.vocab,
                                   target_mode=args.action == "vocab_target")
    else:
        env_cfg = PandaTaskEnvCfg(task=args.task, terminate_on_success=False)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    base = PandaKeypointEnv(env_cfg) if args.task == "keypoints" else PandaSkillEnv(env_cfg) if skills else PandaVocabEnv(env_cfg) if args.action.startswith("vocab") \
        else PandaTaskEnv(env_cfg)
    env = RslRlVecEnvWrapper(base)

    ppo_cfg = VocabPPOCfg if args.action.startswith("vocab") else RawPPOCfg
    agent_cfg = ppo_cfg(max_iterations=args.max_iterations, seed=args.seed, device=env.unwrapped.device)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    default_name = f"keypoints{'_hold' if args.hold_fix else ''}_chain{args.max_chain}_s{args.seed}" if args.task == "keypoints" else \
        f"skills_{args.skill_reward}_chain{args.max_chain}_s{args.seed}" if skills \
        else f"{args.task}_{args.action}_s{args.seed}"
    log_dir = Path("runs/rl") / (args.run_name or default_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / "meta.json"
    sim_s_per_iter = agent_cfg.num_steps_per_env * args.num_envs * env.unwrapped.step_dt
    if not (args.resume and meta_path.exists()):
        meta_path.write_text(json.dumps(
        {"task": args.task, "seed": args.seed, "num_envs": args.num_envs, "action": args.action,
         "num_steps_per_env": agent_cfg.num_steps_per_env, "sim_seconds_per_iteration": sim_s_per_iter,
         "vocab": args.vocab if args.action.startswith("vocab") else None,
         **({"skill_reward": args.skill_reward, "max_chain": args.max_chain, "ref": args.ref, "bank": args.bank}
            if skills else {})}, indent=1))
    if args.resume:
        (log_dir / "resumed.txt").open("a").write(f"resumed from {args.resume}\n")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    remaining = agent_cfg.max_iterations
    if args.resume:
        runner.load(args.resume)
        remaining = agent_cfg.max_iterations - runner.current_learning_iteration
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
