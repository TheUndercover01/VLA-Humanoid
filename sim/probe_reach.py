"""Measure the home TCP pose and the Panda's top-down reach along x (table frame).

Each env drives the TCP to one target (x on a grid, y = 0, at grasp or hover height)
with the workspace clamp lifted, then reports position error and tilt from vertical.

    PYTHONPATH=. ./isaaclab.sh -p sim/probe_reach.py --headless [--video media/reach_probe.mp4 --enable_cameras]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=None)
parser.add_argument("--rot_weight", type=float, default=1.0)
parser.add_argument("--y", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import sim.envs.panda_env as panda_env  # noqa: E402
from sim.envs.panda_env import MAX_DXYZ, PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from sim.video import Recorder  # noqa: E402

XS = [round(-0.3 + 0.025 * i, 3) for i in range(31)]    # -0.30 .. 0.45
ZS = [0.03, 0.14]


def main():
    panda_env.ROT_WEIGHT = args.rot_weight
    targets = torch.tensor([[x, args.y, z] for z in ZS for x in XS])
    cfg = PandaTaskEnvCfg(task="lift", cameras=args.video is not None, image_size=384)
    cfg.scene.num_envs = len(targets)
    cfg.sim.device = args.device
    cfg.episode_length_s = 100.0
    env = PandaTaskEnv(cfg)
    env.ws_high[:] = 1.0
    env.ws_low[:2] = -1.0
    obs, _ = env.reset()
    dev = env.device
    targets = targets.to(dev)
    # the red cube would get in the way of low targets: park it far to the side
    env.red.write_root_pose_to_sim(torch.cat([env.scene.env_origins + torch.tensor([0.5, 0.4, 0.025], device=dev),
                                              torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(env.num_envs, 1)], -1))
    env.blue.write_root_pose_to_sim(torch.cat([env.scene.env_origins + torch.tensor([0.5, -0.4, 0.025], device=dev),
                                               torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(env.num_envs, 1)], -1))

    rec = None
    if args.video:
        # one low target near the reach limit, one comfortable one
        show = [XS.index(0.2), XS.index(0.45)]
        rec = Recorder(args.video, env_ids=show, caption="reach probe: TCP driven to x = 0.20 m (top), 0.45 m (bottom)")

    hold = torch.zeros(env.num_envs, 5, device=dev)
    hold[:, 4] = 1.0
    for _ in range(20):
        obs, *_ = env.step(hold)
        if rec:
            rec.add(obs, "home")
    tcp = env.to_table(env.tcp_w())
    print(f"HOME_TCP (table frame): {tcp.mean(0).tolist()}  yaw {env.tcp_yaw().mean():.3f}", flush=True)

    for t in range(80):
        a = hold.clone()
        a[:, :3] = ((targets - env.cmd) / MAX_DXYZ).clamp(-1, 1)
        obs, *_ = env.step(a)
        if rec:
            rec.add(obs, f"t = {t * 0.1:.1f} s")
    tcp = env.to_table(env.tcp_w())
    err = (tcp - targets).norm(dim=-1)
    tilt = torch.rad2deg(torch.acos((-env.hand_rot()[:, 2, 2]).clamp(-1, 1)))
    q4 = env.robot.data.joint_pos[:, 3]
    print(f"rot_weight {args.rot_weight}, y {args.y}; joint4 upper limit {env.q_high[3]:.3f}", flush=True)
    print("   x      z   pos_err_mm  tilt_deg  joint4", flush=True)
    for i in range(len(targets)):
        x, _, z = targets[i].tolist()
        print(f"{x:6.3f}  {z:5.3f}  {err[i] * 1000:9.1f}  {tilt[i]:8.1f}  {q4[i]:6.3f}", flush=True)
    q = env.robot.data.joint_pos[:, :7]
    print("joint limits low ", [round(v, 3) for v in env.q_low.tolist()], flush=True)
    print("joint limits high", [round(v, 3) for v in env.q_high.tolist()], flush=True)
    for x in [0.25, 0.3, 0.4]:
        i = XS.index(x)
        print(f"q at x={x}, z={ZS[0]}:", [round(v, 3) for v in q[i].tolist()], flush=True)
    if rec:
        rec.save()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
