"""Franka Panda forward kinematics (modified DH, torch) for randomised starting poses (12 Oct).

Frames: the robot base is the environment origin (panda_env.TABLE_ORIGIN = table frame origin in that frame). The hand frame is the
flange rotated -45 deg about z (as panda_hand in the USD), and the TCP is TCP_OFFSET along the hand's z axis, the same definition as
PandaTaskEnv.tcp_w() / hand_rot() / tcp_yaw(), so a pose computed here equals what the simulator reports after the joints are written.
"""
import math

import torch

# (a, d, alpha) of joints 1-7 (Craig's modified DH, Franka documentation); flange: d = 0.107
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
       (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2)]
_FLANGE_D = 0.107


def _mdh(a, d, alpha, theta):
    """Batch of modified-DH transforms: RotX(alpha) TransX(a) RotZ(theta) TransZ(d); theta: (n,)."""
    n = theta.shape[0]
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = torch.cos(theta), torch.sin(theta)
    T = torch.zeros(n, 4, 4, dtype=theta.dtype, device=theta.device)
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = ct, -st, a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = st * sa, ct * sa, ca, ca * d
    T[:, 3, 3] = 1.0
    return T


def hand_pose(q, tcp_offset=0.1034):
    """q: (n, 7) joint angles -> (tcp position (n, 3) in the base frame, hand rotation (n, 3, 3), hand yaw (n,))."""
    n = q.shape[0]
    T = torch.eye(4, dtype=q.dtype, device=q.device).repeat(n, 1, 1)
    for i, (a, d, alpha) in enumerate(_DH):
        T = T @ _mdh(a, d, alpha, q[:, i])
    T = T @ _mdh(0.0, _FLANGE_D, 0.0, torch.zeros(n, dtype=q.dtype, device=q.device))
    c, s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=q.dtype, device=q.device)
    R = T[:, :3, :3] @ Rz
    p = T[:, :3, 3] + R[:, :, 2] * tcp_offset
    return p, R, torch.atan2(R[:, 1, 0], R[:, 0, 0])
