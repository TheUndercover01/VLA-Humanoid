"""Calibrate the fixed camera of the real clips once: focal length and table plane (6 Oct 2026).

The two flat markers (11, 13) lie on a line, so they cannot fix the tilt of the table plane about that line (their own
normals are only good to ~5 degrees), and resting objects came out 6-10 cm above the table. Objects that sit still the whole
clip (the distractor cube or can) are known to rest on the table: cube centre 2.75 cm, can centre 4.25 cm above it. For
each candidate focal length we place the markers, take those resting objects and fit the plane z = a x + b y + c through
them; the focal length with the smallest residual wins, and the table frame is rotated onto the fitted plane.

    python -m handtrack.calibrate_session /home/ayush/vla_stuff/data/clips --out /home/ayush/vla_stuff/data/clips/session_calib.npz
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from handtrack import track_objects as to

REST_Z = {"cube": to.CUBE_SIDE / 2, "can": to.CAN_H / 2}


def clip_paths(root):
    return sorted(p for p in Path(root).glob("*/*.mp4") if p.parent.name not in ("rejected", "candidates"))


def read_dets(path, every):
    cap = cv2.VideoCapture(str(path))
    det = to.detector()
    out, i = [], 0
    w = h = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            out.append(to.detect(det, fr))
            h, w = fr.shape[:2]
        i += 1
    cap.release()
    return out, w, h


def global_T(all_dets, K):
    Ts = []
    for dets in all_dets:
        for m in dets[:15]:
            if to.TARGET_ID in m and to.BIG_ID in m:
                Ts.append(to.table_frame(m, K))
    T = np.mean(Ts, 0)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return T


def static_objects(all_dets, K, T, std_max=0.004, seen_min=0.6):
    """(position, expected resting z, name) of every object that stays put through a clip."""
    pts = []
    for ci, dets in enumerate(all_dets):
        for name, fn in (("cube", lambda m: to.cube_obs(m, K, T)[0]), ("can", lambda m: to.can_obs(m, K, T))):
            P = [fn(m) for m in dets]
            P = np.array([p for p in P if p is not None])
            if len(P) < max(10, seen_min * len(dets)):
                continue
            if P[:, :2].std(0).max() < std_max and P[:, 2].std() < std_max:
                pts.append((np.median(P, 0), REST_Z[name], name, ci))
    return pts


def fit_plane(pts):
    """z - expected = a x + b y + c[class]: one tilt, one vertical offset per object class (cube, can). The offsets absorb the
    marker-mount and marker-size errors, which are constant per object."""
    names = sorted(REST_Z)
    A = np.array([[p[0][0], p[0][1]] + [float(p[2] == n) for n in names] for p in pts])
    z = np.array([p[0][2] - p[1] for p in pts])
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ sol
    return sol, float(np.sqrt((res ** 2).mean())), res


def rotate_plane(T, sol):
    """Rotate/shift the table frame so the fitted plane z = a x + b y + c becomes z = 0."""
    a, b, c = sol
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    s, cth = np.linalg.norm(v), z @ n
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)   # rotation taking z to n
    T2 = np.eye(4)
    T2[:3, :3] = R.T @ T[:3, :3]
    T2[:3, 3] = R.T @ (T[:3, 3] - np.array([0, 0, c]))
    return T2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--f", type=float, default=0, help="fix the focal length (px) instead of searching for the zero-tilt one")
    a = ap.parse_args()
    paths = clip_paths(a.root)
    all_dets, w, h = [], 0, 0
    for p in paths:
        d, w, h = read_dets(p, a.every)
        all_dets.append(d)
    print(f"{len(paths)} clips read", flush=True)

    def at(f):
        K = to.intrinsics(f, w, h)
        T = global_T(all_dets, K)
        pts = static_objects(all_dets, K, T)
        sol, rms, res = fit_plane(pts)
        return K, T, pts, sol, rms

    # the focal length is not observable from objects that all rest on the table, except through the tilt: at the true f the
    # plane of the flat markers 11/13 and the plane of the resting objects are parallel
    best = None
    for f in ([a.f] if a.f else np.arange(900, 1801, 25)):
        K, T, pts, sol, rms = at(f)
        tilt = float(np.hypot(sol[0], sol[1]))
        print(f"f={f:.0f}: {len(pts)} resting objects, tilt a={sol[0]:+.3f} b={sol[1]:+.3f}, offsets {np.round(sol[2:] * 100, 1)} cm, rms {rms * 1000:.1f} mm", flush=True)
        if best is None or tilt < best[0]:
            best = (tilt, f)
    for f in ([] if a.f else np.arange(best[1] - 24, best[1] + 25, 4)):
        K, T, pts, sol, rms = at(f)
        tilt = float(np.hypot(sol[0], sol[1]))
        if tilt < best[0]:
            best = (tilt, f)
    f = float(best[1])
    K, T, pts, sol, rms = at(f)
    T = rotate_plane(T, np.array([sol[0], sol[1], 0.0]))
    pts = static_objects(all_dets, K, T)
    off = {n: float(np.mean([p[0][2] - p[1] for p in pts if p[2] == n])) for n in REST_Z}
    resid = np.array([p[0][2] - p[1] - off[p[2]] for p in pts])
    print(f"f = {f:.0f} px (tilt {np.degrees(np.arctan(best[0])):.2f} deg before the rotation); vertical offsets after the rotation: "
          + ", ".join(f"{n} {v * 100:+.1f} cm" for n, v in off.items()) + f"; residual rms {np.sqrt((resid ** 2).mean()) * 1000:.1f} mm over {len(pts)} resting objects")
    np.savez(a.out, f=f, w=w, h=h, T_cam_table=T, offset_cube=off["cube"], offset_can=off["can"], resid_rms=np.sqrt((resid ** 2).mean()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
