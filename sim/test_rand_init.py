"""12 Oct: check the random starting pose (panda_env rand_init) and the analytic forward kinematics against the simulator.
Same env as the planned teacher: hand-set thresholds (no --derived), hand leg, no scripted starts (bank_p 0).
    VLA_OBJECTS=cube_cylinder scripts/run_isaac.sh <log> sim/test_rand_init.py --headless --device cuda:0
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--rand_init", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.envs.panda_fk import hand_pose  # noqa: E402
from sim.envs.panda_env import TABLE_ORIGIN  # noqa: E402


def make(rand_init):
    cfg = PandaKeypointEnvCfg(task="c1", terminate_on_success=False, max_chain=2, hold_fix=True, action_filter=0.95, hand_leg=True,
                              bank_p=0.0, rand_init=rand_init)
    cfg.ref_path = "data/processed/keypoint_ref_real_straight.npz"
    cfg.bank_path = "data/processed/state_bank_standin.npz"
    cfg.episode_s = 20.0
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    return PandaKeypointEnv(cfg)


def fk_error(env):
    p, R, yaw = hand_pose(env.robot.data.joint_pos[:, env.arm_ids].double())
    sim_tcp = env.tcp_w() - env.table_origin
    fk_tcp = p.float() - torch.tensor(TABLE_ORIGIN, device=env.device)
    return (sim_tcp - fk_tcp).norm(dim=-1)


for ri in (args.rand_init,):
    env = make(ri)
    env.reset()
    zero = torch.zeros(env.num_envs, 5, device=env.device)
    e0 = fk_error(env)
    tcp0 = env.to_table(env.tcp_w())
    print(f"[rand_init {ri}] after reset: FK vs sim TCP error mean {e0.mean()*1000:.1f} mm, max {e0.max()*1000:.1f} mm", flush=True)
    print(f"   start TCP (table frame) x {tcp0[:,0].min():.2f}..{tcp0[:,0].max():.2f}  y {tcp0[:,1].min():.2f}..{tcp0[:,1].max():.2f}  z {tcp0[:,2].min():.2f}..{tcp0[:,2].max():.2f}; std {tcp0.std(0).cpu().numpy().round(3)}", flush=True)
    print(f"   hand pointing down (1 = straight down): min {(-env.hand_rot()[:, 2, 2]).min():.2f}", flush=True)
    vmax, bad = 0.0, 0
    for t in range(40):
        _, rew, _, _, _ = env.step(zero)
        vmax = max(vmax, env.robot.data.joint_vel[:, env.arm_ids].abs().max().item())
        bad += int(not torch.isfinite(env.robot.data.joint_pos).all())
    err = (env.to_table(env.tcp_w()) - env.cmd).norm(dim=-1)
    print(f"   after 40 zero-action ticks: FK error max {fk_error(env).max()*1000:.1f} mm; |TCP - cmd| mean {err.mean()*100:.1f} cm, max {err.max()*100:.1f} cm; "
          f"max joint speed {vmax:.2f} rad/s; non-finite steps {bad}; TCP z min {env.to_table(env.tcp_w())[:,2].min():.2f}", flush=True)
    env.close()
app.close()
