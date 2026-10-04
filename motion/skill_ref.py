"""Clips -> per-skill reference: the only source of the skill reward (sim/envs/skill_env.py).

Each clip is reduced to the fingertip midpoint (ee_pos), the finger closure (grip, 1 = closed)
and the cube (obj_pos). Every skill is described the same way, with no per-skill code:

  h  fingertips relative to the cube               (how the hand moves around it)
  o  cube relative to where the cube ends up       (what the skill does to it)

both in a frame whose x axis is the cube's horizontal displacement if the skill moves the cube
sideways in most clips (push, place-down), otherwise the cube -> fingertips direction at the
start (reach, pick-lift). Each clip's 6-D path x = [h, o] and its grip are resampled to N_PHASE
points and averaged over clips:

  path   (N_PHASE, 6)  mean [h, o] per phase
  sigma  (N_PHASE,)    RMS spread of the clips around the mean per phase (floor SIGMA_FLOOR):
                       the reward is tolerant where the human varied, strict where they did not
  grip   (N_PHASE,)    mean finger closure per phase
  grip_sigma (N_PHASE,) spread of the closure per phase (floor GRIP_FLOOR): with it, progress along
                       the path also tells apart phases where only the fingers move (closing on the cube)
  delta  (3,)          mean cube displacement in the skill frame (where the cube ends, for skills
                       whose command gives no destination: reach ~0, pick-lift = the lift)
  moves  bool          the skill moves the cube sideways (its command needs a destination)
  done_radius          how close to the path's end counts as done: separates the successful
                       clips' end states from failed ones ("_fail" in the file name); with no
                       failures, the largest end deviation among the successes (floor SIGMA_FLOOR;
                       a 2 cm floor let reach end 2 cm above the cube's centre, so pick-lift then
                       closed on the top edge and the cube slipped out)

    python -m motion.skill_ref --clips data/processed/clips_standin --out data/processed/skill_ref_standin.npz
"""
import argparse
from pathlib import Path

import numpy as np

from sim.clips import SKILLS, infer_object, load, validate

N_PHASE = 20
SIGMA_FLOOR = 0.01      # m: below this, differences are tracking noise
GRIP_FLOOR = 0.1        # closure (0..1): below this, differences are tracking noise
MOVE_XY = 0.05          # a clip moves the cube sideways if its end is this far from its start


def rot_z(v, yaw):
    """Express (..., 3) vectors in a frame whose x axis is at angle yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = v[..., 0], v[..., 1]
    return np.stack([c * x + s * y, -s * x + c * y, v[..., 2]], -1)


def skill_moves(clips):
    """True if most clips of a skill move the cube sideways."""
    return bool(np.mean([np.linalg.norm(infer_object(c)[-1, :2] - infer_object(c)[0, :2]) > MOVE_XY
                         for c in clips]) > 0.5)


def frame_yaw(ee0, obj0, obj_end, moves):
    d = obj_end[:2] - obj0[:2] if moves else ee0[:2] - obj0[:2]
    return np.arctan2(d[1], d[0])


def canonical(c, moves):
    """(T, 6) path [h, o] of one clip in its skill frame."""
    ee, obj = np.asarray(c["ee_pos"], float), infer_object(c)
    yaw = frame_yaw(ee[0], obj[0], obj[-1], moves)
    return np.concatenate([rot_z(ee - obj, yaw), rot_z(obj - obj[-1], yaw)], -1)


def resample(x, n=N_PHASE):
    """Resample (T, ...) samples to n points evenly spaced in phase."""
    x = np.asarray(x, float)
    t = np.linspace(0, 1, len(x))
    tq = np.linspace(0, 1, n)
    flat = x.reshape(len(x), -1)
    return np.stack([np.interp(tq, t, flat[:, j]) for j in range(flat.shape[1])], -1).reshape((n,) + x.shape[1:])


def fit_threshold(pos, neg):
    """Done radius from end deviations: the largest success, unless some failures end that close,
    then halfway between the medians of the two groups."""
    hi = float(np.max(pos))
    inside = [n for n in neg if n <= hi]
    if inside:          # failures look like successes: put the line between the two groups' medians
        return float((np.median(pos) + np.median(neg)) / 2)
    return hi


def build(clips, names):
    out = {"skills": np.array(SKILLS), "n_phase": N_PHASE, "sigma_floor": SIGMA_FLOOR, "grip_floor": GRIP_FLOOR}
    for skill in SKILLS:
        sel = [(c, nm) for c, nm in zip(clips, names) if c["skill"] == skill]
        good = [c for c, nm in sel if "_fail" not in nm]
        bad = [c for c, nm in sel if "_fail" in nm]
        if not good:
            raise SystemExit(f"no successful {skill} clips")
        k = skill.replace("-", "_")
        moves = skill_moves(good)
        paths = np.stack([resample(canonical(c, moves)) for c in good])          # (C, P, 6)
        mean = paths.mean(0)
        sigma = np.sqrt(((paths - mean) ** 2).sum(-1).mean(0))
        dev = lambda cs: [np.linalg.norm(canonical(c, moves)[-1] - mean[-1]) for c in cs]  # noqa: E731
        out[f"{k}_path"] = mean.astype(np.float32)
        out[f"{k}_sigma"] = np.maximum(sigma, SIGMA_FLOOR).astype(np.float32)
        grips = np.stack([resample(c["grip"]) for c in good])
        out[f"{k}_grip"] = grips.mean(0).astype(np.float32)
        out[f"{k}_grip_sigma"] = np.maximum(grips.std(0), GRIP_FLOOR).astype(np.float32)
        out[f"{k}_delta"] = (-mean[0, 3:]).astype(np.float32)
        out[f"{k}_moves"] = moves
        out[f"{k}_done_radius"] = max(fit_threshold(dev(good), dev(bad)), SIGMA_FLOOR)
        out[f"{k}_n"], out[f"{k}_n_fail"] = len(good), len(bad)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    clips, names = [], []
    for f in sorted(Path(a.clips).glob("*.npz")):
        c = load(f)
        if validate(c)[0]:
            print(f"skipping {f.name}")
            continue
        clips.append(c)
        names.append(f.name)
    ref = build(clips, names)
    np.savez(a.out, **ref)
    for skill in SKILLS:
        k = skill.replace("-", "_")
        path, sig = ref[f"{k}_path"], ref[f"{k}_sigma"]
        print(f"{skill:10s} {ref[f'{k}_n']} clips ({ref[f'{k}_n_fail']} failed), moves cube: {ref[f'{k}_moves']} | "
              f"hand vs cube {np.round(path[0, :3] * 100, 1)} -> {np.round(path[-1, :3] * 100, 1)} cm | "
              f"cube displacement {np.round(ref[f'{k}_delta'] * 100, 1)} cm | sigma {sig[0] * 100:.1f} -> "
              f"{sig[-1] * 100:.1f} cm | grip {ref[f'{k}_grip'][0]:.2f} -> {ref[f'{k}_grip'][-1]:.2f} | "
              f"done radius {ref[f'{k}_done_radius'] * 100:.1f} cm")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
