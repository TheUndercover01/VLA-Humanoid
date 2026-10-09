"""Approved real clips -> clip npz files (interface contract in CLOUD_PROMPT.md), object tracks from ArUco markers.

    python -m handtrack.make_clips /home/ayush/vla_stuff/data/clips --calib /home/ayush/vla_stuff/data/clips/session_calib.npz \
        --out data/processed/clips_real

Every approved clip of <clips root>/<file>/segments_<file>.csv (rejected/ and candidates/ are not in the csv) becomes one
<skill>_<object>_<dest>_<NN>.npz at 10 Hz in the table frame (handtrack/track_objects.py):
  obj_pos (T, 3) and obj_quat (T, 4, w x y z) of the object the clip moves, from its markers (the cylinder has no yaw: identity),
  t, skill, plus object / dest / source file.
Heights: the object's marker pose gives a centre whose height is only good to a constant per object (marker mount, plane level),
so each clip is anchored to the table: pick-lift and push start with the object resting on the table; place-down ends with it
resting (on the table or target pad, on the can, or on the cube). Gaps up to 0.5 s are interpolated, a clip with more than
30% of its frames unseen is dropped and listed.
ee_pos / ee_yaw / grip are a PROXY for now (the object centre while the object moves, grip 1 once it is above its resting
height), flagged ee_proxy = True: the object-only keypoint reward (motion/keypoint_ref.py) does not use them; the hand tracker
replaces them before the state bank is built from these clips.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from handtrack import track_objects as to

HALF = {"cube": to.CUBE_SIDE / 2, "cylinder": to.CAN_H / 2}
TOP = {"cube": to.CUBE_SIDE, "cylinder": to.CAN_H}
MAX_MISSING = 0.3             # a clip with more of its frames unseen than this is dropped
FILT = 9                      # frames (60 fps) of the moving average, 0.15 s


def rest_height(obj, skill, dest):
    """Centre height of the object when it rests where the clip puts it (place-down) or starts it."""
    if skill == "place-down" and dest in ("cube", "cylinder"):
        return TOP[dest] + HALF[obj]
    return HALF[obj]


def drop_jumps(x, win=15, tol=0.04):
    """NaN out samples more than tol metres from the median of their neighbours (a wrong marker detection)."""
    x = x.copy()
    ok = np.isfinite(x[:, 0])
    idx = np.nonzero(ok)[0]
    for k, i in enumerate(idx):
        near = idx[max(0, k - win):k + win + 1]
        if len(near) >= 5 and np.linalg.norm(x[i] - np.median(x[near], 0)) > tol:
            x[i] = np.nan
    return x


def fill(x, fps, max_gap_s=0.8):
    """Linear interpolation over short gaps of a (T, ...) array with NaN rows; returns (filled, fraction still missing)."""
    x = x.copy()
    ok = np.isfinite(x.reshape(len(x), -1)).all(1)
    if ok.sum() < 2:
        return x, 1.0
    idx = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(idx, idx[ok], flat[ok, j])
    # NaN back where the gap is longer than max_gap_s (also before the first / after the last sighting)
    gap = np.zeros(len(x), bool)
    run = 0
    for i in range(len(x)):
        run = 0 if ok[i] else run + 1
        gap[i] = run / fps > max_gap_s
    for i in range(len(x) - 2, -1, -1):                   # extend a long gap back over its whole run
        if gap[i + 1] and not ok[i]:
            gap[i] = True
    flat[gap] = np.nan
    out = flat.reshape(x.shape)
    lead = np.argmax(ok)
    trail = len(ok) - 1 - np.argmax(ok[::-1])
    out[:lead] = np.nan
    out[trail + 1:] = np.nan
    return out, float(np.isnan(out.reshape(len(out), -1)[:, 0]).mean())


def smooth(x):
    k = np.ones(FILT) / FILT
    pad = np.pad(x, ((FILT // 2, FILT // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def make_clip(path, row, calib):
    """row: dict from segments csv. Returns (clip dict, note) or (None, reason)."""
    obj, skill, dest = row["object"], row["skill"], row["dest"]
    tr = to.track(path, calib=calib)
    fps = tr["fps"]
    pos = tr["cube_pos"] if obj == "cube" else tr["cyl_pos"]
    yaw = tr["cube_yaw"] if obj == "cube" else np.zeros(len(pos))
    pos = drop_jumps(pos)
    pos, miss = fill(pos, fps)
    if miss > MAX_MISSING:
        return None, f"object unseen in {miss:.0%} of the frames"
    yaw_f, _ = fill(yaw[:, None], fps)
    keep = np.isfinite(pos[:, 0])
    pos, yaw_f = pos[keep], yaw_f[keep, 0]
    t = tr["t"][keep]
    pos = smooth(pos)
    yaw_f = np.unwrap(4 * yaw_f) / 4
    yaw_f = smooth(yaw_f[:, None])[:, 0]
    # anchor the height to the table
    n0 = max(3, int(0.4 * fps))
    if skill == "place-down":
        z_off = np.median(pos[-n0:, 2]) - rest_height(obj, skill, dest)
    else:
        z_off = np.median(pos[:n0, 2]) - HALF[obj]
    pos[:, 2] -= z_off
    # 10 Hz
    tq = np.arange(t[0], t[-1], 0.1)
    p10 = np.stack([np.interp(tq, t, pos[:, j]) for j in range(3)], 1)
    y10 = np.interp(tq, t, yaw_f)
    quat = np.stack([np.cos(y10 / 2), np.zeros_like(y10), np.zeros_like(y10), np.sin(y10 / 2)], 1)
    rest = HALF[obj]
    lifted = (p10[:, 2] - rest > 0.01) | (skill == "push")
    clip = dict(t=tq - tq[0], ee_pos=p10.copy(), ee_yaw=y10, grip=lifted.astype(float), obj_pos=p10, obj_quat=quat,
                skill=skill, object=obj, dest=dest, source=str(path.name), ee_proxy=True, focal=tr["focal"])
    note = (f"{len(tq)} samples ({tq[-1] - tq[0]:.1f} s), z anchor {z_off * 100:+.1f} cm, peak height {100 * (p10[:, 2].max() - rest):.1f} cm, "
            f"xy travel {100 * np.linalg.norm(p10[-1, :2] - p10[0, :2]):.1f} cm")
    return clip, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    calib = dict(np.load(a.calib))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    count = {}
    for seg in sorted(Path(a.root).glob("*/segments_*.csv")):
        for row in csv.DictReader(open(seg)):
            if row["skill"] in ("calibrate", "board"):
                continue
            path = seg.parent / row["file"]
            if not path.exists():                             # rejected / moved clip
                continue
            key = f"{row['skill']}_{row['object']}" + (f"_{row['dest']}" if row["skill"] == "place-down" else "")
            clip, note = make_clip(path, row, calib)
            if clip is None:
                print(f"DROP {row['file']}: {note}", flush=True)
                continue
            count[key] = count.get(key, 0) + 1
            name = f"{key}_{count[key]:02d}.npz"
            np.savez(out / name, **clip)
            print(f"{name:34s} <- {row['file']:26s} {note}", flush=True)
    print("clips per kind:", count)


if __name__ == "__main__":
    main()
