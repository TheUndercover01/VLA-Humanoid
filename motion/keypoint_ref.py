"""Clips -> keypoint waypoints: the only source of the keypoint reward (sim/envs/keypoint_env.py).

Only the object is used, never the hand. Each clip's cube pose (obj_pos, obj_quat; from an ArUco
marker on the real cube) gives its 8 corners in the table frame. A clip of one skill becomes a list of
waypoints: the corners' offsets from where the cube ends up (its final pose in the clip), in a frame
whose x axis is the cube's horizontal travel direction (for skills that move it sideways) or the table's
x axis (pick-lift). Stored relative to the goal like this, the waypoints fit any start and destination.

Per skill (push, pick-lift, place-down; reach moves no object and has no waypoints):
  waypoints (N_WP, 8, 3)  mean corner offsets from the goal pose at N_WP evenly spaced phases
  spread    (N_WP,)       RMS corner distance of the clips from the mean at each phase (floor
                          SPREAD_FLOOR): how close counts as reaching that waypoint
  delta     (3,)          mean displacement of the cube centre, start -> end, in the skill frame
                          (for pick-lift, the lift: where the goal is when no destination is given)
  moves     bool          the skill moves the cube sideways in most clips (needs a destination)
  end_grip  float         how the hand ends the skill in the clips (grip, 1 = closed; mean of the last sample):
                          place-down opens, pick-lift stays closed. With --release, a command is only done
                          when the robot's gripper command ends the same way (same side of 0.5)
  end_turn  float         how much the clips turn the cube by the end: 90th percentile over clips of
                          the corner distance between its final pose and the same position with its
                          starting yaw (push turns the cube, lift and place do not); added to the done
                          tolerance, because the goal keeps the starting yaw

    python -m motion.keypoint_ref --clips data/processed/clips_standin --out data/processed/keypoint_ref_standin.npz
"""
import argparse
from pathlib import Path

import numpy as np

from sim.clips import load, validate
from sim.envs.tasks import CUBE_HALF

SKILLS = ["push", "pick-lift", "place-down"]
N_WP = 20
SPREAD_FLOOR = 0.005     # m: below this, corner differences are tracking noise
MOVE_XY = 0.05           # a clip moves the cube sideways if it ends this far from where it started
# the 8 corners of a cube in its own frame
LOCAL_CORNERS = CUBE_HALF * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def quat_to_mat(q):
    """(..., 4) quaternions (w, x, y, z) -> (..., 3, 3) rotation matrices."""
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def corners(pos, quat):
    """(T, 3) positions and (T, 4) quaternions -> (T, 8, 3) corner positions."""
    return np.asarray(pos, float)[:, None] + np.einsum("tij,kj->tki", quat_to_mat(quat), LOCAL_CORNERS)


def rot_z(v, yaw):
    """Express (..., 3) vectors in a frame whose x axis is at angle yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def resample(x, n=N_WP):
    """Resample (T, ...) samples to n points evenly spaced in phase."""
    x = np.asarray(x, float)
    t, tq = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
    flat = x.reshape(len(x), -1)
    return np.stack([np.interp(tq, t, flat[:, j]) for j in range(flat.shape[1])], -1).reshape((n,) + x.shape[1:])


def skill_moves(clips):
    return bool(np.mean([np.linalg.norm(c["obj_pos"][-1, :2] - c["obj_pos"][0, :2]) > MOVE_XY for c in clips]) > 0.5)


def clip_waypoints(c, moves):
    """(N_WP, 8, 3) corner offsets from the clip's final pose, in the skill frame; and the centre's displacement."""
    pos, quat = np.asarray(c["obj_pos"], float), np.asarray(c["obj_quat"], float)
    k = corners(pos, quat)
    d = pos[-1, :2] - pos[0, :2]
    yaw = np.arctan2(d[1], d[0]) if moves else 0.0
    return resample(rot_z(k - k[-1], yaw)), rot_z(pos[-1] - pos[0], yaw)


def end_turn(c):
    """Corner distance between a clip's final pose and that position with the clip's starting yaw."""
    pos, quat = np.asarray(c["obj_pos"], float), np.asarray(c["obj_quat"], float)
    r = quat_to_mat(quat[:1])[0]
    yaw0 = np.arctan2(r[1, 0], r[0, 0])
    end = corners(pos[-1:], quat[-1:])[0]
    same = corners(pos[-1:], np.array([[np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]]))[0]
    return float(np.linalg.norm(end[:, None] - same[None], axis=-1).min(1).mean())


def build(clips):
    out = {"skills": np.array(SKILLS), "n_wp": N_WP, "spread_floor": SPREAD_FLOOR}
    for skill in SKILLS:
        sel = [c for c in clips if c["skill"] == skill]
        if not sel:
            raise SystemExit(f"no {skill} clips")
        moves = skill_moves(sel)
        wps, deltas = zip(*[clip_waypoints(c, moves) for c in sel])
        wps = np.stack(wps)                                                   # (C, N_WP, 8, 3)
        mean = wps.mean(0)
        spread = np.sqrt(((wps - mean) ** 2).sum(-1).mean((0, 2)))            # RMS corner distance per phase
        k = skill.replace("-", "_")
        out[f"{k}_waypoints"] = mean.astype(np.float32)
        out[f"{k}_spread"] = np.maximum(spread, SPREAD_FLOOR).astype(np.float32)
        out[f"{k}_delta"] = np.mean(deltas, 0).astype(np.float32)
        out[f"{k}_moves"] = moves
        out[f"{k}_end_turn"] = float(np.percentile([end_turn(c) for c in sel], 90))
        out[f"{k}_end_grip"] = float(np.mean([np.asarray(c["grip"], float)[-1] for c in sel]))
        out[f"{k}_n"] = len(sel)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    clips = []
    for f in sorted(Path(a.clips).glob("*.npz")):
        c = load(f)
        if validate(c)[0] or "_fail" in f.name or c["skill"] not in SKILLS:
            continue
        if "obj_quat" not in c or np.isnan(np.asarray(c["obj_pos"], float)).any():
            print(f"skipping {f.name}: needs a tracked cube pose (obj_pos and obj_quat)")
            continue
        clips.append(c)
    ref = build(clips)
    np.savez(a.out, **ref)
    for skill in SKILLS:
        k = skill.replace("-", "_")
        wp, sp = ref[f"{k}_waypoints"], ref[f"{k}_spread"]
        centre = wp.mean(1) * 100
        print(f"{skill:10s} {ref[f'{k}_n']} clips, moves sideways: {ref[f'{k}_moves']} | centre offset from goal "
              f"{np.round(centre[0], 1)} -> {np.round(centre[N_WP // 2], 1)} -> {np.round(centre[-1], 1)} cm | "
              f"spread {sp[0] * 100:.1f} -> {sp[N_WP // 2] * 100:.1f} -> {sp[-1] * 100:.1f} cm | "
              f"travel {np.round(ref[f'{k}_delta'] * 100, 1)} cm | end turn {ref[f'{k}_end_turn'] * 100:.1f} cm | "
              f"end grip {ref[f'{k}_end_grip']:.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
