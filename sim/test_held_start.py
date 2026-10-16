"""14 Oct: smoke test of the held start (keypoint_env held_p). Phase 1: every env starts with the object in the hand (random arm pose), the arm
holds still with the fingers commanded closed for 2 s: does the object stay between the fingers? Phase 2: the teacher runs the held-start
episodes (put it down, then pick up the next): commands completed per episode.
    VLA_OBJECTS=cube_cylinder VLA_FIXED_TARGET=0.03,-0.12 scripts/run_isaac.sh <log> sim/test_held_start.py --headless --device cuda:0 --teacher <pt>"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--teacher", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402

cfg = PandaKeypointEnvCfg(task="c1", terminate_on_success=False, max_chain=2, hold_fix=True, action_filter=0.95, hand_leg=True,
                          push_lift_pen=0.3, bank_p=0.0, rand_init=1.0, held_p=1.0, cube_push_only=True)
cfg.ref_path = "data/processed/keypoint_ref_real_straight.npz"
cfg.bank_path = "data/processed/state_bank_standin.npz"
cfg.episode_s = 20.0
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = PandaKeypointEnv(cfg)
import sim.envs.keypoint_env as ke  # noqa: E402
ke.P_STACKED = 0.0
env.reset()
n = env.num_envs
obj = env.cmds[:, 0, 1]
first = env.cmds[:, 0, 0]
print(f"first commands: put-down {int((first == ke.PLACE).sum())}/{n}; object cube {int((obj == 0).sum())}, cylinder {int((obj == 1).sum())}", flush=True)


def gap():
    s = env.state()
    o = torch.where((obj == 1)[:, None], s["blue"], s["red"])
    return (o - s["tcp"]).norm(dim=-1), o[:, 2]


g0, z0 = gap()
a = torch.zeros(n, 5, device=env.device)
a[:, 4] = -1.0
for _ in range(20):
    env.step(a)
g1, z1 = gap()
held = g1 < 0.03
print(f"[hold 2 s] object-to-fingertips at start: mean {g0.mean()*100:.1f} cm; after 2 s: mean {g1.mean()*100:.1f} cm, "
      f"still in hand {int(held.sum())}/{n} (cube {int((held & (obj == 0)).sum())}/{int((obj == 0).sum())}, cylinder {int((held & (obj == 1)).sum())}/{int((obj == 1).sum())}); "
      f"object height {z1.mean()*100:.1f} cm (min {z1.min()*100:.1f})", flush=True)
if args.teacher:
    from sim.rl_cfg import RslPolicy
    env.reset()
    pol = RslPolicy(args.teacher, env)
    best = torch.zeros(n, device=env.device)
    for t in range(200):
        act, _ = pol.act(env)
        env.step(act)
        best = torch.maximum(best, env.completed_final.float())
    nc = env.n_cmd.float()
    first = env.cmds[:, 0, 0]
    print(f"[teacher, 20 s] commands done per episode: mean {best.mean():.2f} of {nc.mean():.1f}; both done {int((best >= nc).sum())}/{n}; "
          f"first (put-down) done {int((best >= 1).sum())}/{n}", flush=True)
env.close()
app.close()
