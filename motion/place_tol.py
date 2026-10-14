"""14 Oct (user: infer the stricter put-down done check from the clips): how precisely do the real objects end up resting where they were put?
For every place-down clip: the object's final height minus its exact resting height on what it was put on (table: its own half height; on the
other object: that object's height + its half height; sizes from sim/envs/tasks.py, the same objects as the clips). The put-down tolerance is the
95th percentile of |that deviation| over all place-down clips (marker noise included), plus the clips' end turn as before (keypoint_env).
    python3 -m motion.place_tol [--clips data/processed/clips_real] [--ref data/processed/keypoint_ref_real_straight.npz]  -> prints the tolerance (m)"""
import argparse
import glob

import numpy as np

CUBE_H, CYL_H = 0.05, 0.08                     # sim/envs/tasks.py: CUBE_HALF 0.025, CYL_H 0.08
HALF = {"cube": CUBE_H / 2, "cylinder": CYL_H / 2}
TOP = {"cube": CUBE_H, "cylinder": CYL_H, "table": 0.0}

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="data/processed/clips_real")
ap.add_argument("--ref", default="data/processed/keypoint_ref_real_straight.npz")
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
dev = []
for f in sorted(glob.glob(a.clips + "/*.npz")):
    c = np.load(f, allow_pickle=True)
    if str(c["skill"]) != "place-down":
        continue
    obj, dest = str(c["object"]), str(c["dest"])
    if obj not in HALF or dest not in TOP:
        continue
    d = float(c["obj_pos"][-1, 2] - (TOP[dest] + HALF[obj]))
    dev.append(d)
    if a.verbose:
        print(f"{f.split('/')[-1]:32s} {obj} on {dest}: final height off its resting height by {d * 100:+.1f} cm")
dev = np.abs(np.array(dev))
end_turn = float(np.load(a.ref)["place_down_end_turn"])
tol = float(np.percentile(dev, 95)) + end_turn
if a.verbose:
    print(f"{len(dev)} place-down clips: |deviation| median {np.median(dev) * 100:.2f} cm, p95 {np.percentile(dev, 95) * 100:.2f} cm, max {dev.max() * 100:.2f} cm; "
          f"+ end turn {end_turn * 100:.2f} cm -> put-down tolerance {tol * 100:.2f} cm (was 2.14 cm)")
print(f"{tol:.4f}")
