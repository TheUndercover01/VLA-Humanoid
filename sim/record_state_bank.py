"""Replay every skill clip in sim (headless) and save the robot and cube state at every tick.

RL episodes in the skill env can start from any of these states (reference state
initialisation, as in DeepMimic): the arm part-way through a reach, the cube already in the
hand at the start of a carry, and so on. The states come from the clips, played on the robot.

    PYTHONPATH=. ./isaaclab.sh -p sim/record_state_bank.py --headless \
        --clips data/processed/clips_standin --out data/processed/state_bank_standin.npz
Writes per frame: joint_pos (9), cube_pos (3), cube_quat (4), cmd (3), cmd_yaw, opening,
skill (index into sim.clips.SKILLS), phase in [0, 1), clip index, the clip's own cube position
(clip_cube: frames where the sim cube is far from it, e.g. dropped in the replay, are unusable); plus each clip's end
cube position (the destination of a push or place-down).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clips", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.clips import PRE_TICKS, SKILLS, infer_object, layout_for, load, preroll_plan, validate  # noqa: E402
from sim.envs.panda_env import MAX_DXYZ, MAX_DYAW, PandaTaskEnv, PandaTaskEnvCfg  # noqa: E402


def main():
    clips = []
    for f in sorted(Path(args.clips).glob("*.npz")):
        c = load(f)
        if validate(c)[0] or "_fail" in f.name:        # failed attempts are not start states
            continue
        c["obj_pos"] = infer_object(c)
        clips.append(c)
    n = len(clips)
    rng = np.random.default_rng(args.seed)
    cfg = PandaTaskEnvCfg(task="c1", terminate_on_success=False)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaTaskEnv(cfg)
    env.set_layouts(torch.tensor(np.stack([layout_for(c, rng) for c in clips])))
    env.reset()
    dev = env.device

    pre = torch.tensor(np.stack([preroll_plan(c) for c in clips]), device=dev)
    yaw0 = torch.tensor([float(c["ee_yaw"][0]) for c in clips], device=dev)
    for t in range(PRE_TICKS):
        a = torch.zeros(n, 5, device=dev)
        a[:, :3] = ((pre[:, t, :3] - env.cmd) / MAX_DXYZ).clamp(-1, 1)
        a[:, 3] = ((yaw0 - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)
        a[:, 4] = pre[:, t, 3] * 2 - 1
        env.step(a)

    lens = [len(c["t"]) for c in clips]
    T = max(lens)
    ee = torch.zeros(n, T, 3, device=dev)
    yaw = torch.zeros(n, T, device=dev)
    opening = torch.zeros(n, T, device=dev)
    for i, c in enumerate(clips):
        k = lens[i]
        ee[i, :k] = torch.tensor(c["ee_pos"], device=dev)
        ee[i, k:] = ee[i, k - 1]
        yaw[i, :k] = torch.tensor(np.unwrap(c["ee_yaw"]), device=dev)
        yaw[i, k:] = yaw[i, k - 1]
        opening[i, :k] = 1.0 - torch.tensor(c["grip"], device=dev)
        opening[i, k:] = opening[i, k - 1]
    frames = []
    for t in range(T - 1):
        frames.append({"joint_pos": env.robot.data.joint_pos.clone(), "cube_pos": env.to_table(env.red.data.root_pos_w),
                       "cube_quat": env.red.data.root_quat_w.clone(), "cmd": env.cmd.clone(),
                       "cmd_yaw": env.cmd_yaw.clone(), "opening": env.grip_target / 0.04})
        a = torch.zeros(n, 5, device=dev)
        a[:, :3] = ((ee[:, t + 1] - env.cmd) / MAX_DXYZ).clamp(-1, 1)
        a[:, 3] = ((yaw[:, t + 1] - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)
        a[:, 4] = opening[:, t + 1] * 2 - 1
        env.step(a)
    end_cube = env.to_table(env.red.data.root_pos_w).cpu().numpy()

    out = {k: [] for k in list(frames[0]) + ["skill", "phase", "clip", "clip_cube"]}
    for i, c in enumerate(clips):
        k = lens[i] - 1
        for t in range(k):
            for key in frames[0]:
                out[key].append(frames[t][key][i].cpu().numpy())
            out["skill"].append(SKILLS.index(c["skill"]))
            out["phase"].append(t / k)
            out["clip"].append(i)
            out["clip_cube"].append(c["obj_pos"][t].astype(np.float32))
    bank = {k: np.array(v, np.float32 if k not in ("skill", "clip") else np.int64) for k, v in out.items()}
    bank["clip_end_cube"] = end_cube.astype(np.float32)
    bank["clip_skill"] = np.array([SKILLS.index(c["skill"]) for c in clips])
    np.savez(args.out, **bank)
    for j, s in enumerate(SKILLS):
        m = bank["skill"] == j
        off = np.linalg.norm(bank["cube_pos"][m] - bank["clip_cube"][m], axis=1)
        print(f"{s:10s} {m.sum():5d} frames, sim cube within 3 cm of the clip's in {(off < 0.03).mean():.0%}, cube z {bank['cube_pos'][m, 2].min() * 100:.1f}-"
              f"{bank['cube_pos'][m, 2].max() * 100:.1f} cm", flush=True)
    print(f"wrote {len(bank['skill'])} frames from {n} clips to {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
