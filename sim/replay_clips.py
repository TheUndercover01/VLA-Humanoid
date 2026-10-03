"""Replay skill clips in sim with cameras: the image/state/action data for the B1 SmolVLA fine-tune.

Each env replays one clip. A scripted pre-roll first sets up the clip's starting situation
(a clip that starts holding the cube picks it up first; one that starts at the cube lowers
the open gripper around it), then each 10 Hz tick commands the clip's next pose with the
raw 5-D action, which is what gets recorded. Prompts are atomic only (vla/prompts.py).

    PYTHONPATH=. ./isaaclab.sh -p sim/replay_clips.py --headless --enable_cameras \
        --clips data/processed/clips_standin --out data/processed/replays/standin
Writes one npz per clip: front, wrist (T, H, W, 3) uint8, state (T, 6), action (T, 5), prompt, skill.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clips", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.clips import PRE_TICKS, infer_object, layout_for, load, preroll_plan, validate  # noqa: E402
from sim.envs.panda_env import MAX_DXYZ, MAX_DYAW, PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402
from vla.prompts import ATOMIC_PROMPTS  # noqa: E402

def main():
    files, clips = [], []
    for f in sorted(Path(args.clips).glob("*.npz")):
        c = load(f)
        errs, _ = validate(c)
        if errs:
            print(f"skipping {f.name}: {'; '.join(errs)}", flush=True)
            continue
        c["obj_pos"] = infer_object(c)       # fills untracked samples
        files.append(f)
        clips.append(c)
    n = len(clips)
    rng = np.random.default_rng(args.seed)
    cfg = PandaTaskEnvCfg(task="c1", cameras=True, image_size=args.image_size, terminate_on_success=False)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaTaskEnv(cfg)
    env.set_layouts(torch.tensor(np.stack([layout_for(c, rng) for c in clips])))
    obs, _ = env.reset()
    dev = env.device

    pre = torch.tensor(np.stack([preroll_plan(c) for c in clips]), device=dev)
    yaw0 = torch.tensor([float(c["ee_yaw"][0]) for c in clips], device=dev)
    for t in range(PRE_TICKS):
        a = torch.zeros(n, 5, device=dev)
        a[:, :3] = ((pre[:, t, :3] - env.cmd) / MAX_DXYZ).clamp(-1, 1)
        a[:, 3] = ((yaw0 - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)
        a[:, 4] = pre[:, t, 3] * 2 - 1
        obs, *_ = env.step(a)

    lens = [len(c["t"]) for c in clips]
    T = max(lens)
    ee = torch.zeros(n, T, 3, device=dev)
    yaw = torch.zeros(n, T, device=dev)
    opening = torch.zeros(n, T, device=dev)
    for i, c in enumerate(clips):         # pad short clips with their last pose
        k = lens[i]
        ee[i, :k] = torch.tensor(c["ee_pos"], device=dev)
        ee[i, k:] = ee[i, k - 1]
        yaw[i, :k] = torch.tensor(np.unwrap(c["ee_yaw"]), device=dev)
        yaw[i, k:] = yaw[i, k - 1]
        opening[i, :k] = 1.0 - torch.tensor(c["grip"], device=dev)
        opening[i, k:] = opening[i, k - 1]
    rec = {k: [] for k in ["front", "wrist", "state", "action"]}
    start_red = env.state()["red"].clone()
    for t in range(T - 1):
        s = env.state()
        a = torch.zeros(n, 5, device=dev)
        a[:, :3] = ((ee[:, t + 1] - env.cmd) / MAX_DXYZ).clamp(-1, 1)
        a[:, 3] = ((yaw[:, t + 1] - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)
        a[:, 4] = opening[:, t + 1] * 2 - 1
        ty = env.tcp_yaw()
        rec["state"].append(torch.cat([s["tcp"], torch.sin(ty)[:, None], torch.cos(ty)[:, None],
                                       s["grip"][:, None] / 0.08], -1).cpu())
        rec["action"].append(a.cpu())
        rec["front"].append(obs["front"][..., :3].to(torch.uint8).cpu())
        rec["wrist"].append(obs["wrist"][..., :3].to(torch.uint8).cpu())
        obs, *_ = env.step(a)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = env.state()
    moved = (end["red"] - start_red).norm(dim=-1).cpu()
    track = (env.to_table(env.tcp_w()) - ee[:, -1]).norm(dim=-1).cpu()
    stats = {}
    for i, (f, c) in enumerate(zip(files, clips)):
        k = lens[i] - 1
        skill = str(c["skill"])
        np.savez_compressed(out / f.name, **{key: torch.stack(v, 1)[i, :k].numpy() for key, v in rec.items()},
                            prompt=ATOMIC_PROMPTS[skill], skill=skill)
        st = stats.setdefault(skill, {"n": 0, "end_err": [], "red_moved": [], "red_z": []})
        st["n"] += 1
        st["end_err"].append(float(track[i]))
        st["red_moved"].append(float(moved[i]))
        st["red_z"].append(float(end["red"][i, 2]))
    for skill, st in stats.items():
        print(f"{skill:10s} {st['n']} clips | TCP vs clip end {np.mean(st['end_err']) * 1000:5.1f} mm | "
              f"red moved {np.mean(st['red_moved']) * 100:5.1f} cm | red z at end {np.mean(st['red_z']) * 100:4.1f} cm",
              flush=True)
    print(f"wrote {n} replays to {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
