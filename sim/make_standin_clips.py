"""Stand-in for the phone clips: atomic skill clips cut from scripted-expert episodes.

Writes data/processed/clips_standin/<skill>_<idx>.npz in the clip interface format
(t, ee_pos, ee_yaw, grip, obj_pos, obj_quat, skill; table frame, 10 Hz; obj_quat is the cube's
orientation (w, x, y, z), which the phone clips get from an ArUco marker on the cube). Each clip shows one skill:
  reach      home -> gripper around the cube           (from lift episodes)
  pick-lift  close on the cube and lift it              (from lift episodes)
  place-down carry the held cube, lower it, open        (from c1 episodes)
  push       approach behind the cube and push it       (from push episodes)
Layouts come from a training seed, never the eval states.

    PYTHONPATH=. ./isaaclab.sh -p sim/make_standin_clips.py --headless --source lift   # reach, pick-lift
    ... --source c1                                                                     # place-down
    ... --source push                                                                   # push
(One source task per process: Isaac Lab cannot open a second sim in the same process.)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, choices=["lift", "c1", "push"])
parser.add_argument("--per_skill", type=int, default=50)
parser.add_argument("--out", default="data/processed/clips_standin")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from sim.expert import SKILLS, ScriptedExpert  # noqa: E402

DT = 0.1
# (source task, which expert skill labels form the clip)
SOURCES = {"reach": ("lift", ["reach"]), "pick-lift": ("lift", ["pick-lift"]),
           "place-down": ("c1", ["place-down"]), "push": ("push", ["reach", "push"])}


def record(task, n, seed):
    """Run the expert on n training layouts; return per-env lists of per-step records."""
    cfg = PandaTaskEnvCfg(task=task)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaTaskEnv(cfg)
    env.set_layouts(tasks.sample_layouts(task, n, torch.Generator().manual_seed(seed)))
    env.reset()
    ex = ScriptedExpert()
    ex.reset(env)
    logs = [[] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    ok = torch.zeros(n, dtype=torch.bool, device=env.device)
    for _ in range(int(env.max_episode_length)):
        s = env.state()
        a, skill = ex.act(env)
        snap = (s["tcp"].cpu().numpy(), env.tcp_yaw().cpu().numpy(), ((1 - a[:, 4]) / 2).cpu().numpy(),
                s["red"].cpu().numpy(), skill.cpu().numpy(), env.red.data.root_quat_w.cpu().numpy())
        for i in range(n):
            if not done[i]:
                logs[i].append([x[i] for x in snap])
        env.step(a)
        ex.update(env)
        new = env.episode_done & ~done
        ok |= new & env.last_episode["success"].bool()
        done |= new
        if done.all():
            break
    env.close()
    return [lg for lg, good in zip(logs, ok.tolist()) if good]


def cut(log, labels, first_only=True):
    """Indices of the first contiguous run of steps whose skill label is in labels."""
    idx = [k for k, step in enumerate(log) if SKILLS[step[4]] in labels]
    if not idx:
        return None
    run = [idx[0]]
    for k in idx[1:]:
        if k != run[-1] + 1:
            break
        run.append(k)
    return run


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = record(args.source, args.per_skill + 10, args.seed + ["lift", "c1", "push"].index(args.source))
    for skill, (task, labels) in SOURCES.items():
        if task != args.source:
            continue
        count = 0
        for log in logs:
            run = cut(log, labels)
            if run is None or len(run) < 3:
                continue
            steps = [log[k] for k in run]
            if run[-1] + 1 < len(log):          # include the end pose, but not the next skill's gripper command
                end = list(log[run[-1] + 1])
                end[2] = steps[-1][2]
                steps.append(end)
            np.savez(out / f"{skill}_{count:03d}.npz",
                     t=np.arange(len(steps)) * DT,
                     ee_pos=np.stack([s[0] for s in steps]).astype(np.float32),
                     ee_yaw=np.array([s[1] for s in steps], np.float32),
                     grip=np.array([s[2] for s in steps], np.float32),
                     obj_pos=np.stack([s[3] for s in steps]).astype(np.float32),
                     obj_quat=np.stack([s[5] for s in steps]).astype(np.float32),
                     skill=skill)
            count += 1
            if count == args.per_skill:
                break
        lens = [len(np.load(p)["t"]) for p in out.glob(f"{skill}_*.npz")]
        print(f"{skill}: {count} clips, length {min(lens)}-{max(lens)} steps", flush=True)


if __name__ == "__main__":
    main()
    app.close()
