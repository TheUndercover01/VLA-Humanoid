"""MuJoCo scene (Franka Panda, table, cubes, target) and a Cartesian controller.

World frame = table frame from the interface contract: origin at the table
centre, x forward (away from the robot), y left, z up, table top at z = 0.
The robot base sits at x = BASE_X.
"""
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets" / "franka_emika_panda"
BASE_X = -0.5
CUBE_HALF = 0.025
TCP_OFFSET = 0.1034           # hand frame -> fingertip centre along hand z
HOME_Q = np.array([0, 0.2, 0, -2.0, 0, 2.2, -0.785])
HOME_TCP = np.array([0.0, 0.0, 0.25])


def build_model(with_cameras=True):
    spec = mujoco.MjSpec.from_file(str(ASSETS / "panda.xml"))
    spec.option.timestep = 0.002
    spec.body("link0").pos = [BASE_X, 0, 0]
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005] * 3)
    wb = spec.worldbody

    wb.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                pos=[0, 0, -0.02], rgba=[0.55, 0.55, 0.55, 1])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.45, 0.01],
                pos=[0.0, 0, -0.01], rgba=[0.85, 0.80, 0.70, 1], friction=[1.0, 0.01, 0.001])

    for name, rgba, pos in [("red", [0.9, 0.1, 0.1, 1], [0.3, 0.1, CUBE_HALF]),
                            ("blue", [0.1, 0.2, 0.9, 1], [0.3, -0.1, CUBE_HALF])]:
        b = wb.add_body(name=name, pos=pos)
        b.add_freejoint(name=f"{name}_free")
        b.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[CUBE_HALF] * 3, rgba=rgba, mass=0.05,
                   friction=[1.5, 0.02, 0.002], condim=4)

    tg = wb.add_body(name="target", mocap=True, pos=[0.3, 0.0, 0.001])
    tg.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.001, 0],
                rgba=[0.1, 0.8, 0.2, 1], contype=0, conaffinity=0)

    if with_cameras:
        wb.add_camera(name="front", pos=[0.9, 0.0, 0.7], xyaxes=[0, 1, 0, -0.6, 0, 0.8])
        spec.body("hand").add_camera(name="wrist", pos=[0.05, 0, 0.04],
                                     xyaxes=[0, -1, 0, -1, 0, 0], fovy=75)
    return spec.compile()


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def target_rot(yaw):
    """Tool z pointing down, rotated by yaw about world z."""
    flip = np.diag([1.0, -1.0, -1.0])
    return rot_z(yaw) @ flip


class Panda:
    """Wraps model/data with a damped-least-squares Cartesian pose controller."""

    def __init__(self, model, data, control_dt=0.1):
        self.m, self.d = model, data
        self.substeps = int(round(control_dt / model.opt.timestep))
        self.dt = control_dt
        self.site = model.site("tcp").id
        self.arm_dofs = np.arange(7)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.grip_ctrl = 255.0   # open
        self.peak_force = 0.0

    def tcp_pos(self):
        return self.d.site_xpos[self.site].copy()

    def tcp_rot(self):
        return self.d.site_xmat[self.site].reshape(3, 3).copy()

    def tcp_yaw(self):
        r = self.tcp_rot()
        # yaw of tool x axis projected on the table plane (flip keeps x unchanged)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def grip_opening(self):
        return float(self.d.qpos[7] + self.d.qpos[8])   # 0 closed .. 0.08 open

    def reset(self, tcp=HOME_TCP, yaw=0.0, opening=1.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:7] = HOME_Q
        self.d.qpos[7:9] = 0.04
        self.d.ctrl[:7] = HOME_Q
        self.set_grip(opening)
        mujoco.mj_forward(self.m, self.d)
        for _ in range(60):
            self.servo(np.array(tcp), yaw)
            self.step_physics(self.substeps // 5)
        self.peak_force = 0.0

    def set_grip(self, opening):
        """opening in [0, 1]: 1 open, 0 closed."""
        self.grip_ctrl = 255.0 * float(np.clip(opening, 0.0, 1.0))
        self.d.ctrl[7] = self.grip_ctrl

    def servo(self, tcp_target, yaw_target, max_step=0.08):
        """One DLS IK step toward the target pose; sets joint position targets."""
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
        J = np.vstack([self.jacp[:, :7], self.jacr[:, :7]])
        e_p = tcp_target - self.tcp_pos()
        r, r_des = self.tcp_rot(), target_rot(yaw_target)
        e_r = 0.5 * sum(np.cross(r[:, i], r_des[:, i]) for i in range(3))
        err = np.concatenate([e_p * 2.0, e_r])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
        dq = np.clip(dq, -max_step, max_step)
        q = self.d.qpos[:7]
        self.d.ctrl[:7] = np.clip(q + dq, self.m.actuator_ctrlrange[:7, 0], self.m.actuator_ctrlrange[:7, 1])

    def step_physics(self, n):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
        f = np.zeros(6)
        peak = 0.0
        for i in range(self.d.ncon):
            mujoco.mj_contactForce(self.m, self.d, i, f)
            peak = max(peak, abs(f[0]))
        self.peak_force = max(self.peak_force, peak)

    def move_to(self, tcp_target, yaw_target, opening):
        """Command a pose for one control step (control_dt of physics)."""
        self.set_grip(opening)
        per = self.substeps // 5
        for _ in range(5):
            self.servo(np.asarray(tcp_target), yaw_target)
            self.step_physics(per)
