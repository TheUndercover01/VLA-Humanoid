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
parser.add_argument("--action", default="raw", choices=["raw", "vocab"])
parser.add_argument("--vocab", default="data/processed/vocab_standin.pt")
parser.add_argument("--resume", default=None, help="checkpoint to continue from (same run dir)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from sim.envs.vocab_env import PandaVocabEnv, PandaVocabEnvCfg  # noqa: E402
from sim.rl_cfg import RawPPOCfg, VocabPPOCfg  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    if args.action == "vocab":
        env_cfg = PandaVocabEnvCfg(task=args.task, terminate_on_success=False, vocab_path=args.vocab)
    else:
        env_cfg = PandaTaskEnvCfg(task=args.task, terminate_on_success=False)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    base = PandaVocabEnv(env_cfg) if args.action == "vocab" else PandaTaskEnv(env_cfg)
    env = RslRlVecEnvWrapper(base)

    ppo_cfg = VocabPPOCfg if args.action == "vocab" else RawPPOCfg
    agent_cfg = ppo_cfg(max_iterations=args.max_iterations, seed=args.seed, device=env.unwrapped.device)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    log_dir = Path("runs/rl") / (args.run_name or f"{args.task}_{args.action}_s{args.seed}")
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / "meta.json"
    sim_s_per_iter = agent_cfg.num_steps_per_env * args.num_envs * env.unwrapped.step_dt
    if not (args.resume and meta_path.exists()):
        meta_path.write_text(json.dumps(
        {"task": args.task, "seed": args.seed, "num_envs": args.num_envs, "action": args.action,
         "num_steps_per_env": agent_cfg.num_steps_per_env, "sim_seconds_per_iteration": sim_s_per_iter,
         "vocab": args.vocab if args.action == "vocab" else None}, indent=1))
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
