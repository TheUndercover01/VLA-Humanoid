"""Phase 0 gate: measure a stock Isaac-Lift-Cube-Franka-v0 rsl_rl checkpoint.

Runs one 5 s episode on N envs and reports how often the cube is lifted and how
close it ends to the commanded goal.

    ./isaaclab.sh -p /path/to/sim/isaac_check_lift.py --headless --checkpoint model_1499.pt
    ./isaaclab.sh -p /path/to/sim/isaac_check_lift.py --headless --pretrained
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--pretrained", action="store_true", help="NVIDIA's published checkpoint")
parser.add_argument("--num_envs", type=int, default=1024)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

TASK = "Isaac-Lift-Cube-Franka-v0"
RSL_VERSION = metadata.version("rsl-rl-lib")


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_VERSION)
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg))
    agent_cfg.device = env.unwrapped.device
    ckpt = get_published_pretrained_checkpoint("rsl_rl", TASK) if args.pretrained else args.checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    ckpt = handle_deprecated_rsl_rl_checkpoint(ckpt, RSL_VERSION)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    cube, robot = u.scene["object"], u.scene["robot"]
    n_steps = int(u.max_episode_length)
    obs = env.get_observations()
    ever_lifted = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    lifted_frac = torch.zeros(u.num_envs, device=u.device)
    with torch.inference_mode():
        for t in range(n_steps - 1):          # stop before the time-out reset
            obs, _, _, _ = env.step(policy(obs))
            # stock reward's "lifted" threshold; the cube spawns at 5.5 cm, so skip settling
            lifted = (cube.data.root_pos_w[:, 2] > 0.04) & (t >= 25)
            ever_lifted |= lifted
            lifted_frac += lifted.float()
        goal_b = u.command_manager.get_command("object_pose")[:, :3]
        cube_b = cube.data.root_pos_w - robot.data.root_pos_w   # robot base is unrotated
        dist = torch.norm(goal_b - cube_b, dim=1)
    print(f"checkpoint: {ckpt}", flush=True)
    print(f"envs {u.num_envs}, steps {n_steps - 1}", flush=True)
    print(f"ever lifted > 4 cm:          {ever_lifted.float().mean():.3f}", flush=True)
    print(f"fraction of time lifted:     {(lifted_frac / (n_steps - 26)).mean():.3f}", flush=True)
    print(f"final dist to goal < 5 cm:   {(dist < 0.05).float().mean():.3f}", flush=True)
    print(f"final dist to goal median:   {dist.median():.3f} m", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
