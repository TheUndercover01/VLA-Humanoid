"""Gymnasium tasks on the Panda scene: atomic skills and combined tasks C1-C3.

Raw action (5): dx, dy, dz, dyaw, grip in [-1, 1]. The commanded end-effector
pose is integrated, so one step moves the target by at most MAX_DXYZ metres.
Observations are state-based (poses in the table frame); cameras are only
rendered by the VLA pipeline.
"""
import gymnasium as gym
import mujoco
import numpy as np

from sim.scene import CUBE_HALF, HOME_TCP, Panda, build_model

MAX_DXYZ = 0.04
MAX_DYAW = 0.35
WS_LOW = np.array([-0.10, -0.32, 0.015])
WS_HIGH = np.array([0.42, 0.32, 0.40])
C3_FAR_X = (0.46, 0.52)   # red starts beyond comfortable grasp reach (see scene reach notes)
STAGES = {
    "push": ["reach", "push"],
    "lift": ["reach", "grasp", "lift"],
    "c1": ["reach", "grasp", "lift", "transport", "place"],
    "c2": ["reach", "grasp", "lift", "transport", "place"],
    "c3": ["push", "reach", "grasp", "lift", "transport", "place"],
}
PROMPTS = {
    "push": "push the red cube to the green target",
    "lift": "pick up the red cube",
    "c1": "pick up the red cube and place it on the green target",
    "c2": "stack the red cube on the blue cube",
    "c3": "push the red cube closer, then place it on the green target",
}
SUCCESS_HOLD = 5          # control steps (0.5 s)
EPISODE_STEPS = 150


def sample_layout(task, rng):
    """Random cube / target positions (table frame) for a task."""
    def far(p, others, d=0.12):
        return all(np.linalg.norm(p - o) > d for o in others)

    while True:
        red = rng.uniform([0.05, -0.25], [0.30, 0.25])
        if task == "c3":
            red = np.array([rng.uniform(*C3_FAR_X), rng.uniform(-0.15, 0.15)])
        blue = rng.uniform([0.05, -0.25], [0.30, 0.25])
        tgt = rng.uniform([0.05, -0.25], [0.30, 0.25])
        if task == "push":                      # push target further out than the cube
            tgt = red + rng.uniform([0.08, -0.08], [0.15, 0.08])
        ok = far(red, [blue, tgt]) and (task == "push" or far(blue, [tgt]))
        if ok and np.all(np.abs(tgt) < [0.40, 0.28]):
            return red, blue, tgt


class PandaTaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="c1", seed=0, cameras=False):
        assert task in STAGES, task
        self.task = task
        self.model = build_model(with_cameras=cameras)
        self.data = mujoco.MjData(self.model)
        self.robot = Panda(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        self.red_adr = self.model.joint("red_free").qposadr[0]
        self.blue_adr = self.model.joint("blue_free").qposadr[0]
        self.red_id = self.model.body("red").id
        self.blue_id = self.model.body("blue").id
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (21,), np.float32)
        self.layout = None

    # ---------- state helpers ----------
    def red(self):
        return self.data.xpos[self.red_id].copy()

    def blue(self):
        return self.data.xpos[self.blue_id].copy()

    def target(self):
        return self.data.mocap_pos[0].copy()

    def goal(self):
        """3D position where the red cube's centre should end up."""
        if self.task == "c2":
            return self.blue() + np.array([0, 0, 2 * CUBE_HALF])
        t = self.target()
        return np.array([t[0], t[1], CUBE_HALF])

    def obs(self):
        r = self.robot
        tcp, red, blue, tgt = r.tcp_pos(), self.red(), self.blue(), self.goal()
        yaw = r.tcp_yaw()
        o = np.concatenate([tcp, [np.sin(yaw), np.cos(yaw)], [r.grip_opening() / 0.08],
                            red, blue, self.target(), red - tcp, tgt - red])
        return o.astype(np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None, layout=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        layout = layout if layout is not None else sample_layout(self.task, self.rng)
        self.layout = layout
        red, blue, tgt = layout
        self.robot.reset()
        quat = [1, 0, 0, 0]
        self.data.qpos[self.red_adr:self.red_adr + 7] = [*red, CUBE_HALF, *quat]
        self.data.qpos[self.blue_adr:self.blue_adr + 7] = [*blue, CUBE_HALF, *quat]
        self.data.mocap_pos[0] = [*tgt, 0.001]
        mujoco.mj_forward(self.model, self.data)
        self.robot.step_physics(40)           # let the cubes settle
        self.cmd = self.robot.tcp_pos()
        self.cmd_yaw = self.robot.tcp_yaw()
        self.cmd_open = 1.0
        self.t = 0
        self.hold = 0
        self.reached = {s: False for s in STAGES[self.task]}
        self.start_red = self.red()
        self.start_blue = self.blue()
        self.tcp_hist = [self.robot.tcp_pos()]
        self.robot.peak_force = 0.0
        self.knocked = False
        return self.obs(), {}

    def apply_cmd(self, pos, yaw, opening):
        """Drive the end effector toward a commanded pose for one control step."""
        pos = np.clip(pos, WS_LOW, WS_HIGH)
        tcp = self.robot.tcp_pos()
        self.cmd = tcp + np.clip(pos - tcp, -0.06, 0.06)   # never run away from the arm
        self.cmd_yaw, self.cmd_open = yaw, opening
        self.robot.move_to(self.cmd, yaw, opening)
        self.tcp_hist.append(self.robot.tcp_pos())

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        pos = self.cmd + a[:3] * MAX_DXYZ
        yaw = self.cmd_yaw + a[3] * MAX_DYAW
        self.apply_cmd(pos, yaw, (a[4] + 1) / 2)
        return self._finish_step()

    def _finish_step(self):
        self.t += 1
        self._update_stages()
        ok = self.success()
        self.hold = self.hold + 1 if ok else 0
        success = self.hold >= SUCCESS_HOLD
        red = self.red()
        if red[2] < -0.05 or np.any(np.abs(red[:2]) > 0.5):
            self.knocked = True
        truncated = self.t >= EPISODE_STEPS
        terminated = success or self.knocked
        info = {"success": success, "stage": self.failure_stage() if (terminated or truncated) and not success else None,
                "reward_terms": None}
        if terminated or truncated:
            info.update(self.metrics())
        return self.obs(), self.reward(success), terminated, truncated, info

    # ---------- success, stages, reward ----------
    def red_speed(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.red_id, v, 0)
        return float(np.linalg.norm(v[3:]))

    def success(self):
        red, goal = self.red(), self.goal()
        open_ok = self.robot.grip_opening() > 0.04
        if self.task == "lift":
            return red[2] > 0.10
        if self.task == "push":
            return np.linalg.norm(red[:2] - goal[:2]) < 0.03 and red[2] < 0.04
        rest = self.red_speed() < 0.03
        if self.task == "c2":
            return (np.linalg.norm(red[:2] - goal[:2]) < 0.025 and abs(red[2] - goal[2]) < 0.012
                    and rest and open_ok)
        return np.linalg.norm(red[:2] - goal[:2]) < 0.03 and red[2] < 0.04 and rest and open_ok

    def _update_stages(self):
        r, red, goal = self.robot, self.red(), self.goal()
        tcp = r.tcp_pos()
        near = np.linalg.norm(tcp - red) < 0.06
        lifted_z = red[2] > CUBE_HALF + 0.03
        s = self.reached
        if "push" in s and np.linalg.norm(red[:2] - self.start_red[:2]) > 0.05:
            s["push"] = True
        if "push" in s and red[0] < C3_FAR_X[0] - 0.12:
            s["push"] = True
        if near:
            s["reach"] = True
        if s.get("reach") and near and lifted_z and r.grip_opening() < 0.07:
            s["grasp"] = True
        if s.get("grasp") and red[2] > CUBE_HALF + 0.06:
            s["lift"] = True
        if s.get("lift") and np.linalg.norm(red[:2] - goal[:2]) < 0.06:
            s["transport"] = True
        if self.success():
            for k in s:
                s[k] = True

    def failure_stage(self):
        if self.knocked:
            return "knocked"
        for st in STAGES[self.task]:
            if not self.reached[st]:
                return st
        return "place"

    def reward(self, success):
        r, red, goal = self.robot, self.red(), self.goal()
        tcp = r.tcp_pos()
        d_reach = np.linalg.norm(tcp - (red + [0, 0, 0.01]))
        held = (red[2] > CUBE_HALF + 0.02) and d_reach < 0.05 and r.grip_opening() < 0.07
        d_goal = np.linalg.norm(red[:2] - goal[:2])
        rew = 0.5 * (1 - np.tanh(8 * d_reach))
        if self.task == "push":
            rew = 0.3 * (1 - np.tanh(8 * d_reach)) + 1.5 * (1 - np.tanh(6 * d_goal))
        elif self.task == "lift":
            rew += 2.0 * held + 4.0 * np.clip(red[2] - CUBE_HALF, 0, 0.1) / 0.1 * held
        else:
            if self.task == "c3":
                far = max(red[0] - 0.30, 0.0)       # distance still beyond comfortable reach
                rew += 1.0 * (1 - np.tanh(10 * far))
            rew += 2.0 * held + 3.0 * held * (1 - np.tanh(5 * d_goal))
            at_goal = d_goal < 0.04 and abs(red[2] - goal[2]) < 0.015
            rew += 4.0 * at_goal * (r.grip_opening() / 0.08)
        rew -= 0.01 * float(np.sum(np.square(self.cmd - tcp)))
        return float(rew + (20.0 if success else 0.0))

    def metrics(self):
        h = np.array(self.tcp_hist)
        jerk = np.abs(np.diff(h, n=3, axis=0)).mean() / (self.robot.dt ** 3) if len(h) > 4 else 0.0
        return {"time": self.t * self.robot.dt, "jerk": float(jerk),
                "peak_force": float(self.robot.peak_force)}
