"""Video of how one phone clip becomes the reward: the clip with the ArUco markers found in every frame and the object's tracked
path, next to the same path in the table frame (relative to where the object ends) and the straight start -> goal template the
keypoint reward uses (motion/keypoint_ref.py --straight, tolerance 3 cm). One clip per skill, back to back.

    python -m analysis.clip_video --out media/site/real_clip_to_template.mp4 \
        pick-lift_cube_01 place-down_cube_cylinder_01 push_cube_01
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from handtrack.track_objects import CUBE_IDS, CYL_ID, detector  # noqa: E402

CLIPS = Path("data/processed/clips_real")
RAW = Path("/home/ayush/vla_stuff/data/clips")
REF = Path("data/processed/keypoint_ref_real_straight.npz")
TOL = 0.03
H = 540


def side_view(pos, end, skill):
    """(along, up) of each sample relative to the end pose: along = horizontal distance in the direction of travel."""
    rel = pos - end
    d = rel[0, :2]
    u = -d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 0.02 else np.array([1.0, 0.0])
    return np.stack([rel[:, :2] @ u, rel[:, 2]], -1)


def panel(sv, k, tpl, title, size):
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    a0, a1 = tpl[0], np.zeros(2)
    n = np.array([-(a1 - a0)[1], (a1 - a0)[0]]) / (np.linalg.norm(a1 - a0) + 1e-9)
    band = np.array([a0 + TOL * n, a1 + TOL * n, a1 - TOL * n, a0 - TOL * n])
    ax.fill(band[:, 0] * 100, band[:, 1] * 100, color="tab:green", alpha=0.15, label="template, 3 cm tolerance")
    ax.plot([a0[0] * 100, 0], [a0[1] * 100, 0], "--", color="tab:green", lw=2, label="straight start -> goal (mean of the clips)")
    ax.plot(sv[:k + 1, 0] * 100, sv[:k + 1, 1] * 100, "o-", color="tab:orange", ms=4, label="this clip, tracked marker")
    ax.plot([0], [0], "k*", ms=14, label="goal (where the object ends)")
    lim = np.abs(np.concatenate([sv, tpl])).max() * 100 + 5
    ax.set_xlim(-lim, 6); ax.set_ylim(-lim * 0.6 if sv[:, 1].min() < -0.02 else -6, lim * 0.6 if sv[:, 1].max() > 0.02 else 6)
    ax.set_xlabel("horizontal, cm (travel direction)"); ax.set_ylabel("height, cm")
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = np.load(REF)
    det = detector()
    frames = []
    for name in a.clips:
        c = np.load(CLIPS / f"{name}.npz", allow_pickle=True)
        skill, obj, src = str(c["skill"]), str(c["object"]), str(c["source"])
        tpl = ref[skill.replace("-", "_") + "_waypoints"].mean(1)                  # (20, 3) template centre, relative to the goal
        tpl = np.stack([-np.linalg.norm(tpl[:, :2], axis=-1), tpl[:, 2]], -1)
        pos = c["obj_pos"]
        ok = np.isfinite(pos[:, 0])
        pos = np.where(ok[:, None], pos, np.nan)
        for i in range(1, len(pos)):                     # hold the last seen position where the marker is hidden
            if not ok[i]:
                pos[i] = pos[i - 1]
        end = pos[-1]
        sv = side_view(pos, end, skill)
        raw = next(RAW.glob(f"*/{src}"))
        cap = cv2.VideoCapture(str(raw))
        fps = cap.get(cv2.CAP_PROP_FPS)
        trail = []
        ids_obj = CUBE_IDS if obj == "cube" else (CYL_ID,)
        for k, t in enumerate(c["t"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ret, img = cap.read()
            if not ret:
                break
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                for cc, i in zip(corners, ids.ravel()):
                    if i in ids_obj:
                        trail.append(cc[0].mean(0))
                        break
            for p, q in zip(trail[:-1], trail[1:]):
                cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (0, 165, 255), 6)
            img = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))
            cv2.putText(img, f"phone clip: {skill} the {obj}  ({src})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            right = panel(sv, k, tpl, f"{skill}: what the reward keeps from this clip", (720, H))
            frames.append(np.concatenate([img, right], 1))
        frames += [frames[-1]] * 10                     # hold the last frame 1 s
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.replace(".mp4", "_raw.avi")
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        w.write(cv2.resize(f, (frames[0].shape[1], frames[0].shape[0])))
    w.release()
    print(tmp, len(frames))


if __name__ == "__main__":
    main()
