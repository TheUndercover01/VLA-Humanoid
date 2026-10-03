"""Skill clips (interface contract in CLOUD_PROMPT.md): validation, cube inference, replay set-up.

Pure numpy, no Isaac imports, so it runs anywhere:
    python -m sim.clips data/processed/clips          # validate a folder of clips
"""
import sys
from pathlib import Path

import numpy as np

from sim.envs import tasks
from sim.envs.tasks import BASE_XY, CUBE_HALF, HOME_TCP

SKILLS = ["reach", "push", "pick-lift", "place-down"]
KEYS = ["t", "ee_pos", "ee_yaw", "grip", "obj_pos", "skill"]
REACH_XY = (0.30, 0.79)        # TCP distance from the robot base the arm can serve (sim/probe_reach.py, with slack)
HOVER, GRASP_Z = 0.14, 0.026
PRE_TICKS = 40
PUSH_LEAD = 0.035              # cube centre ahead of the TCP while pushing (half cube + finger)


def load(path):
    c = dict(np.load(path, allow_pickle=True))
    c["skill"] = str(c["skill"])
    return c


def validate(c):
    """Problems with one clip: (errors, warnings). Errors make the clip unusable."""
    err, warn = [], []
    missing = [k for k in KEYS if k not in c]
    if missing:
        return [f"missing keys {missing}"], warn
    t, ee = np.asarray(c["t"]), np.asarray(c["ee_pos"])
    n = len(t)
    for k, shape in [("ee_pos", (n, 3)), ("ee_yaw", (n,)), ("grip", (n,)), ("obj_pos", (n, 3))]:
        if np.asarray(c[k]).shape != shape:
            err.append(f"{k} shape {np.asarray(c[k]).shape}, expected {shape}")
    if err:
        return err, warn
    if c["skill"] not in SKILLS:
        err.append(f"unknown skill {c['skill']!r}")
    if n < 5:
        err.append(f"only {n} samples")
    dt = np.diff(t)
    if np.any(dt <= 0):
        err.append("t is not increasing")
    elif abs(np.median(dt) - 0.1) > 0.02:
        err.append(f"sample period {np.median(dt):.3f} s, expected 0.1 s (10 Hz)")
    if not np.all(np.isfinite(ee)) or not np.all(np.isfinite(c["ee_yaw"])) or not np.all(np.isfinite(c["grip"])):
        err.append("NaN/inf in ee_pos, ee_yaw or grip")
        return err, warn
    if np.abs(ee).max() > 1.5:
        err.append(f"|ee_pos| up to {np.abs(ee).max():.2f}: not metres in the table frame?")
    g = np.asarray(c["grip"])
    if g.min() < -0.01 or g.max() > 1.01:
        err.append(f"grip outside [0, 1]: {g.min():.2f}..{g.max():.2f}")
    r = np.linalg.norm(ee[:, :2] - np.array(BASE_XY), axis=1)
    if r.min() < REACH_XY[0] or r.max() > REACH_XY[1]:
        warn.append(f"TCP {r.min():.2f}..{r.max():.2f} m from the base, arm serves {REACH_XY[0]}..{REACH_XY[1]} m "
                    f"(the replay clamps it)")
    if ee[:, 2].min() < -0.005:
        warn.append(f"TCP goes {-ee[:, 2].min() * 100:.1f} cm below the table top")
    if ee[:, 2].max() > 0.40:
        warn.append(f"TCP up to z = {ee[:, 2].max():.2f} m, workspace top is 0.40 m")
    step = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    if step.max() > 0.08:
        warn.append(f"TCP jumps {step.max() * 100:.1f} cm in one sample (tracking glitch?); the controller moves <= 4 cm")
    obj = np.asarray(c["obj_pos"], float)
    if np.isnan(obj).all():
        warn.append("obj_pos untracked: cube position will be inferred from the gripper motion")
    elif np.isnan(obj).any():
        warn.append(f"obj_pos partly NaN ({np.isnan(obj[:, 0]).mean():.0%}): gaps will be inferred")
    return err, warn


def infer_object(c):
    """Cube positions for samples where obj_pos is NaN, from the gripper motion and the skill."""
    ee = np.asarray(c["ee_pos"], float)
    obj = np.asarray(c["obj_pos"], float).copy()
    if not np.isnan(obj).any():
        return obj
    n, skill = len(ee), c["skill"]
    closed = np.asarray(c["grip"]) > 0.5
    guess = np.zeros((n, 3))
    if skill == "reach":                          # the reach ends at the cube
        guess[:] = [ee[-1, 0], ee[-1, 1], CUBE_HALF]
    elif skill == "pick-lift":                    # on the table until the gripper closes, then in the hand
        k = int(np.argmax(closed)) if closed.any() else n - 1
        guess[:] = [ee[k, 0], ee[k, 1], CUBE_HALF]
        guess[k:] = ee[k:] - (ee[k, 2] - CUBE_HALF) * np.array([0, 0, 1.0])
    elif skill == "place-down":                   # in the hand until the gripper opens, then resting there
        k = int(np.argmax(~closed)) if (~closed).any() else n
        if k > 0:                                 # held: the lowest point of the carry puts it on the table
            guess[:k] = ee[:k] - np.array([0, 0, ee[:k, 2].min() - CUBE_HALF])
        if k < n:
            guess[k:] = [ee[k, 0], ee[k, 1], CUBE_HALF]
    elif skill == "push":                         # ahead of the fingers along the push while they are low
        low = np.nonzero(ee[:, 2] < 0.06)[0]
        if len(low) < 2:
            guess[:] = [ee[-1, 0], ee[-1, 1], CUBE_HALF]
        else:
            a, b = low[0], low[-1]
            d = ee[b, :2] - ee[a, :2]
            u = d / (np.linalg.norm(d) + 1e-9)
            guess[:, :2] = ee[a, :2] + u * PUSH_LEAD
            guess[a:b + 1, :2] = ee[a:b + 1, :2] + u * PUSH_LEAD
            guess[b + 1:, :2] = ee[b, :2] + u * PUSH_LEAD
            guess[:, 2] = CUBE_HALF
    nan = np.isnan(obj).any(axis=1)
    obj[nan] = guess[nan]
    return obj


def preroll_plan(c):
    """(PRE_TICKS, 4) targets x, y, z, opening that set up a clip's starting situation in sim."""
    p0 = np.asarray(c["ee_pos"][0], float)
    holding = infer_object(c)[0, 2] > CUBE_HALF + 0.01
    home = np.array(HOME_TCP)
    if holding:                                  # pick the cube up from under the start pose
        plan = [(p0[0], p0[1], HOVER, 1.0)] * 12 + [(p0[0], p0[1], GRASP_Z, 1.0)] * 8
        plan += [(p0[0], p0[1], GRASP_Z, 0.0)] * 8 + [(*p0, 0.0)] * 12
    elif np.linalg.norm(p0 - home) > 0.02:       # lower the open gripper to the start pose
        plan = [(p0[0], p0[1], HOVER, 1.0)] * 14 + [(*p0, 1.0)] * 26
    else:
        plan = [(*home, 1.0)] * PRE_TICKS
    return np.array(plan[:PRE_TICKS], np.float32)


def layout_for(c, rng):
    """Red where the clip starts it (under the gripper if held); blue and target out of the way."""
    obj = infer_object(c)
    red = c["ee_pos"][0][:2] if obj[0, 2] > CUBE_HALF + 0.01 else obj[0, :2]
    path = np.concatenate([np.asarray(c["ee_pos"])[:, :2], obj[:, :2]])
    lo, hi = np.array(tasks.REGION[0]), np.array(tasks.REGION[1])
    push = c["skill"] == "push"
    for _ in range(1000):
        blue = rng.uniform(lo, hi)
        tgt = obj[-1, :2] if push else rng.uniform(lo, hi)
        far = np.linalg.norm(path - blue, axis=1).min() > 0.10 and np.linalg.norm(blue - tgt) > 0.12
        if far and (push or np.linalg.norm(path - tgt, axis=1).min() > 0.10):
            return np.concatenate([red, blue, tgt]).astype(np.float32)
    raise RuntimeError("no layout keeps blue and the target clear of this clip")


def main(folder):
    files = sorted(Path(folder).glob("*.npz"))
    if not files:
        sys.exit(f"no .npz clips in {folder}")
    bad, counts = 0, {}
    for f in files:
        c = load(f)
        errs, warns = validate(c)
        counts[c.get("skill", "?")] = counts.get(c.get("skill", "?"), 0) + 1
        bad += bool(errs)
        for m in errs:
            print(f"ERROR {f.name}: {m}")
        for m in warns:
            print(f"warn  {f.name}: {m}")
    print(f"{len(files)} clips ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}), {bad} unusable")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/clips")
