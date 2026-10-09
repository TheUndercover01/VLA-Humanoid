"""Object tracking in real phone clips from ArUco markers (6 Oct 2026, the user's setup).

Markers (DICT_4X4_50, black square side in metres):
    0   top of the cylinder (can), 4.25 cm        1  top of the cube, 4.25 cm        2-5  the cube's four sides, 4.25 cm
    11  the target, left in the image, 8 cm       13 big calibration marker, right in the image, 14.5 cm
Cube side 5.5 cm; can diameter 7.6 cm, height 8.5 cm.

Table frame (sim/envs/tasks.py: x away from the robot, y left, z up, table top z = 0): the robot sits beyond the far edge of
the video facing the camera, so x points toward the camera (down in the image) and y toward the image right. The frame is
built from the two static markers 11 and 13 lying flat on the table: z = their mean normal, y from 11 to 13 projected on the
table plane, origin at the midpoint of 11 and 13 (the table centre is unknown; the clips are used relative to the cube
anyway). Camera: pinhole with the principal point at the image centre and the focal length found so that the normals of the
two flat markers agree.

    python -m handtrack.track_objects <clip.mp4> [--out clip_objects.npz] [--debug_dir dir]

Writes t (s, video time), cube_pos (T, 3), cube_yaw (T,) (mod 90 degrees, [-45, 45) deg), cyl_pos (T, 3), cube_seen /
cyl_seen (T,) bool, target_xy (2,), table pose, focal length. Positions are NaN where no marker of that object was seen.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

SIZE = {0: 0.0425, 1: 0.0425, 2: 0.0425, 3: 0.0425, 4: 0.0425, 5: 0.0425, 11: 0.08, 13: 0.145}
CUBE_IDS, CYL_ID, TARGET_ID, BIG_ID = (1, 2, 3, 4, 5), 0, 11, 13
CUBE_SIDE, CAN_H = 0.055, 0.085


def detector(relaxed=True):
    """relaxed (6 Oct): the can's lid marker with a hand over part of it, and tilted markers, were missed by the default
    parameters (Cylinder_lift_07 30% -> 79% of the frames seen); wrong detections are removed by make_clips (jump filter)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if relaxed:
        p.polygonalApproxAccuracyRate = 0.06
        p.maxErroneousBitsInBorderRate = 0.6
        p.errorCorrectionRate = 1.0
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMax = 63
        p.adaptiveThreshWinSizeStep = 6
        p.perspectiveRemovePixelPerCell = 6
        p.minOtsuStdDev = 2.0
    return cv2.aruco.ArucoDetector(d, p)


def square_pts(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)   # aruco corner order: TL, TR, BR, BL


def marker_pose(corners, size, K):
    """(R, t) of the marker in the camera frame; z points out of the marker toward the camera."""
    ok, rvec, tvec = cv2.solvePnP(square_pts(size), corners.reshape(4, 2).astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    if R[:, 2] @ t > 0:                                   # normal pointing away from the camera: flip about the marker's x axis
        R = R @ np.diag([1.0, -1.0, -1.0])
    return R, t


def intrinsics(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def detect(det, frame):
    c, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return {} if ids is None else {int(i): cc for i, cc in zip(ids.flatten(), c)}


def estimate_focal(frames_markers, w, h):
    """Focal length (px) at which the normals of the static flat markers 11 and 13 agree best, over several frames."""
    best = (1e9, None)
    for f in np.arange(800, 3200, 25):
        K = intrinsics(f, w, h)
        errs = []
        for m in frames_markers:
            if TARGET_ID in m and BIG_ID in m:
                a, b = marker_pose(m[TARGET_ID], SIZE[TARGET_ID], K), marker_pose(m[BIG_ID], SIZE[BIG_ID], K)
                if a and b:
                    errs.append(np.degrees(np.arccos(np.clip(a[0][:, 2] @ b[0][:, 2], -1, 1))))
        if errs and np.mean(errs) < best[0]:
            best = (np.mean(errs), f)
    return best[1], best[0]


def table_frame(m, K):
    """4x4 camera -> table transform from the static markers 11 and 13 in one frame's detections."""
    a, b = marker_pose(m[TARGET_ID], SIZE[TARGET_ID], K), marker_pose(m[BIG_ID], SIZE[BIG_ID], K)
    z = a[0][:, 2] * SIZE[TARGET_ID] ** 2 + b[0][:, 2] * SIZE[BIG_ID] ** 2           # normals toward the camera, weighted by area
    z /= np.linalg.norm(z)
    y = b[1] - a[1]
    y = y - (y @ z) * z
    y /= np.linalg.norm(y)                                # from the target (11) to the big marker (13) = image right = robot's left
    x = np.cross(y, z)                                    # right-handed: x toward the camera, down in the image
    R = np.stack([x, y, z], 1)                            # table axes in the camera frame
    o = (a[1] + b[1]) / 2
    # table plane through the marker centres: z = 0 there; the marker sheets are thin so the table top is z = 0
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ o
    return T


def to_table(T, p):
    return (T @ np.append(p, 1))[:3]


def cube_obs(m, K, T):
    """Cube centre and yaw (mod 90 degrees) from every visible cube marker, area-weighted."""
    cs, ws, yaws = [], [], []
    for i in CUBE_IDS:
        if i in m:
            pose = marker_pose(m[i], SIZE[i], K)
            if pose is None:
                continue
            R, t = pose
            n = T[:3, :3] @ R[:, 2]                       # outward normal in the table frame
            centre = to_table(T, t) - n * CUBE_SIDE / 2
            cs.append(centre)
            ws.append(cv2.contourArea(m[i].reshape(4, 2).astype(np.float32)))
            h = n[:2] if abs(n[2]) < 0.7 else (T[:3, :3] @ R[:, 0])[:2]      # side face: its normal; top face: its x axis
            yaws.append(np.arctan2(h[1], h[0]))
    if not cs:
        return None, None
    w = np.array(ws) / np.sum(ws)
    ang = np.array(yaws) * 4                              # 90 degree symmetry: average on the 4x circle
    yaw = np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) / 4
    return (np.array(cs) * w[:, None]).sum(0), yaw


def can_obs(m, K, T):
    if CYL_ID not in m:
        return None
    pose = marker_pose(m[CYL_ID], SIZE[CYL_ID], K)
    if pose is None:
        return None
    R, t = pose
    n = T[:3, :3] @ R[:, 2]
    return to_table(T, t) - n * CAN_H / 2


def track(path, debug_dir=None, every=1, calib=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = detector()
    frames, dets = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            dets.append(detect(det, fr))
            if debug_dir and i == 0:
                frames.append(fr)
        i += 1
    cap.release()
    both = [m for m in dets if TARGET_ID in m and BIG_ID in m]
    if calib is not None:                                 # session calibration (handtrack/calibrate_session.py): fixed camera
        f, err = float(calib["f"]), float("nan")
        K = intrinsics(f, w, h)
        T = np.asarray(calib["T_cam_table"], float)
    else:
        if len(both) < 3:
            raise RuntimeError(f"{path}: markers 11 and 13 together in only {len(both)} frames")
        f, err = estimate_focal([both[k] for k in np.linspace(0, len(both) - 1, min(30, len(both))).astype(int)], w, h)
        K = intrinsics(f, w, h)
        Ts = [table_frame(m, K) for m in both[:min(60, len(both))]]
        T = np.mean(Ts, 0)
        U, _, Vt = np.linalg.svd(T[:3, :3])
        T[:3, :3] = U @ Vt
    n = len(dets)
    cube_pos, cube_yaw, cyl_pos = np.full((n, 3), np.nan), np.full(n, np.nan), np.full((n, 3), np.nan)
    for k, m in enumerate(dets):
        c, yaw = cube_obs(m, K, T)
        if c is not None:
            cube_pos[k], cube_yaw[k] = c, yaw
        p = can_obs(m, K, T)
        if p is not None:
            cyl_pos[k] = p
    tgt = to_table(T, marker_pose(both[0][TARGET_ID], SIZE[TARGET_ID], K)[1])[:2] if both else np.full(2, np.nan)
    out = dict(t=np.arange(n) * every / fps, cube_pos=cube_pos, cube_yaw=cube_yaw, cyl_pos=cyl_pos,
               cube_seen=np.isfinite(cube_pos[:, 0]), cyl_seen=np.isfinite(cyl_pos[:, 0]), target_xy=tgt, T_cam_table=T,
               focal=f, focal_normal_err_deg=err, fps=fps / every)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    o = track(a.clip, every=a.every)
    print(f"{a.clip}: {len(o['t'])} frames, f={o['focal']:.0f} px (normal disagreement {o['focal_normal_err_deg']:.2f} deg), "
          f"cube seen {o['cube_seen'].mean():.0%}, can seen {o['cyl_seen'].mean():.0%}, target (table) {np.round(o['target_xy'], 3)}")
    if a.out:
        np.savez_compressed(a.out, **o)


if __name__ == "__main__":
    main()
