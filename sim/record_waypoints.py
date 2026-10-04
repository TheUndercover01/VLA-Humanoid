"""Record the waypoint paths the keypoint env lays down at every command start, and where the cube went.

Runs PandaKeypointEnv with its training resets (random layouts, starts from the clips' states, chains
of up to 2 commands) and the scripted follower moving the cubes; every time a command starts (episode
reset or the previous command done), its placed waypoints are saved with the cube's track during it.
For analysis/waypoint_video.py.

    PYTHONPATH=. ./isaaclab.sh -p sim/record_waypoints.py --headless --out runs/viz/waypoints.npz
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/viz/waypoints.npz")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--ticks", type=int, default=450)
parser.add_argument("--seed", type=int, default=21)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.keypoint_follower import Follower  # noqa: E402


def main():
    cfg = PandaKeypointEnvCfg(task="c1", terminate_on_success=False, max_chain=2, seed=args.seed)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = PandaKeypointEnv(cfg)
    starts = []                               # one dict per command start
    original = env.start_commands

    def start_commands(ids):                  # record what the env lays down, then let it run on
        original(ids)
        skill, cube, dest = env.command()
        s = env.state()
        ap, aq, ac, op, *_ = env.objects()
        for i in ids.tolist():
            starts.append({"env": i, "step": len(track), "skill": int(skill[i]), "cube": int(cube[i]),
                           "dest": int(dest[i]), "path": env.path[i].cpu().numpy(),
                           "goal": env.goal_corners[i].cpu().numpy(), "start": ac[i].cpu().numpy(),
                           "other": op[i].cpu().numpy(), "target": s["target"][i].cpu().numpy(),
                           "tol": env.wp_tol[skill[i]].cpu().numpy()})

    track, reached, cur, eps = [], [], [], []
    env.start_commands = start_commands
    env.reset()
    fol = Follower(env)
    for _ in range(args.ticks):
        env.step(fol.act(env))
        ap = env.objects()[0]
        track.append(ap.cpu().numpy())
        reached.append(env.wp_reached.cpu().numpy())
        cur.append(env.cur.cpu().numpy())
        eps.append(env.episode_length_buf.cpu().numpy())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = starts[0].keys()
    np.savez(out, **{f"start_{k}": np.stack([np.asarray(st[k]) for st in starts]) for k in keys},
             track=np.stack(track, 1), reached=np.stack(reached, 1), cur=np.stack(cur, 1), ep_len=np.stack(eps, 1),
             template_place=env.wp[3].cpu().numpy(), template_pick=env.wp[2].cpu().numpy(),
             template_push=env.wp[1].cpu().numpy())
    print(f"wrote {len(starts)} command starts from {args.num_envs} envs over {args.ticks} ticks -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
