"""Isaac Lab env shared by all tasks: Panda, table, red/blue cubes, green target, optional cameras.

Raw action (5): dx, dy, dz, dyaw, grip in [-1, 1], at 10 Hz. The commanded TCP pose is
integrated (one step moves it at most MAX_DXYZ) and tracked with damped-least-squares IK
at every physics step. Observations are state-based, in the table frame (see tasks.py).
The task only changes the reset distribution, reward and success check (tasks.py).
"""
from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from sim.envs import tasks
from sim.envs.tasks import CUBE_HALF

TABLE_ORIGIN = (0.5, 0.0, 0.0)          # table frame origin in the env frame (robot base at 0)
TCP_OFFSET = 0.1034                      # panda_hand -> fingertip centre, along hand z
HOME_Q = [0.0, -0.2, 0.0, -2.4, 0.0, 2.2, 0.785]   # tuned in sim/probe_reach.py
HOME_TCP = tasks.HOME_TCP                # TCP at HOME_Q, table frame (measured in sim/probe_reach.py)
ARM_DAMPING = 30.0
MAX_DXYZ = 0.04
MAX_DYAW = 0.35
ROT_WEIGHT = 1.0                         # IK weight of orientation error (position weight is 2)
CTRL_SUBSTEPS = 10                       # physics steps per 10 Hz control tick (sim dt 0.01 s)
MAX_LEAD = 0.06                          # command never runs further than this ahead of the TCP
WS_LOW = (-0.25, -0.32, 0.015)
WS_HIGH = (0.32, 0.32, 0.40)
# near front view: ~1 m from the cube region (env frame x 0.38-0.68, |y| <= 0.2) instead of 1.3 m from a point near
# the robot base, aimed 12 cm above the table so the lifted gripper stays in frame; cubes ~1.6x larger in the image
NEAR_FRONT = ((1.20, 0.38, 0.68), (0.50, 0.0, 0.12))
BASE_XY = (-TABLE_ORIGIN[0], 0.0)        # robot base in the table frame
REACH = (0.32, 0.77)                     # TCP distance from the base (xy) where the gripper stays vertical


def _cube(name, rgba):
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(2 * CUBE_HALF,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16,
                                                         max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgba),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, CUBE_HALF)),
    )


def _cylinder(name, rgba):
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=tasks.CYL_R, height=tasks.CYL_H, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16,
                                                         max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgba),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, tasks.HALF_Z[1])),
    )


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) of an OpenGL camera (looks along -z, y up) at pos looking at target."""
    f = torch.tensor(target, dtype=torch.float32) - torch.tensor(pos, dtype=torch.float32)
    f = f / f.norm()
    x = torch.linalg.cross(f, torch.tensor(up, dtype=torch.float32))
    x = x / x.norm()
    y = torch.linalg.cross(x, f)
    return tuple(quat_from_matrix(torch.stack([x, y, -f], dim=1)).tolist())


@configclass
class PandaTaskEnvCfg(DirectRLEnvCfg):
    task: str = "c1"
    cameras: bool = False
    terminate_on_success: bool = True     # eval: end once success has held; RL training: keep going
    image_size: int = 256
    front_view: str = "far"               # "near" (5 Oct, user): front camera closer to the workspace, see NEAR_FRONT
    episode_s: float = 0.0                # episode length; 0 = tasks.EPISODE_S[task]

    episode_length_s = 15.0               # overwritten per task from tasks.EPISODE_S
    decimation = 10
    action_space = 5
    observation_space = 21
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=10)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=True)

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            joint_pos={**{f"panda_joint{i + 1}": q for i, q in enumerate(HOME_Q)}, "panda_finger_joint.*": 0.04}),
    )
    # The stock high-PD damping (80) with the 12 Nm wrist torque limit caps the wrist at 0.15 rad/s,
    # which throttles every Cartesian move to ~6 cm/s. Lower damping, same stiffness.
    robot.actuators["panda_shoulder"].damping = ARM_DAMPING
    robot.actuators["panda_forearm"].damping = ARM_DAMPING
    if tasks.CUBE_CYL:       # object 0 a white cube, object 1 a red cylinder (the real can is red)
        red = _cube("Red", (0.92, 0.92, 0.92))
        blue = _cylinder("Blue", (0.8, 0.1, 0.1))
    else:
        red = _cube("Red", (0.9, 0.1, 0.1))
        blue = _cube("Blue", (0.1, 0.2, 0.9))
    target = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        # a 6 cm square pad: its corners are the destination keypoints of the keypoint reward
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.2, 0.001)),
    )
    contact = ContactSensorCfg(prim_path="/World/envs/env_.*/(Red|Blue)", history_length=10)
    # cameras use the OpenGL convention (look along -z, y up); same poses as the MuJoCo prototype
    front_cam = TiledCameraCfg(
        prim_path="/World/envs/env_.*/FrontCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.35, 0.55, 0.8), rot=_look_at((1.35, 0.55, 0.8), (0.35, 0.0, 0.15)),
                                        convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 5.0)),
        width=256, height=256,
    )
    wrist_cam = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCam",
        # on the hand, behind the fingers, looking at the space between the fingertips
        offset=TiledCameraCfg.OffsetCfg(pos=(0.08, 0.0, 0.0),
                                        rot=_look_at((0.08, 0.0, 0.0), (0.0, 0.0, 0.14), up=(-1.0, 0.0, 0.0)),
                                        convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=9.0, clipping_range=(0.01, 3.0)),
        width=256, height=256,
    )


class PandaTaskEnv(DirectRLEnv):
    cfg: PandaTaskEnvCfg

    def __init__(self, cfg: PandaTaskEnvCfg, render_mode=None, **kwargs):
        assert cfg.task in tasks.TASKS, cfg.task
        cfg.episode_length_s = cfg.episode_s or tasks.EPISODE_S[cfg.task]
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.task = cfg.task
        self.hand_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.jac_idx = self.hand_idx - 1                 # fixed base: no jacobian for the root
        self.arm_ids = self.robot.find_joints("panda_joint.*")[0]
        self.finger_ids = self.robot.find_joints("panda_finger_joint.*")[0]
        self.q_low = self.robot.data.soft_joint_pos_limits[0, self.arm_ids, 0]
        self.q_high = self.robot.data.soft_joint_pos_limits[0, self.arm_ids, 1]
        self.table_origin = self.scene.env_origins + torch.tensor(TABLE_ORIGIN, device=dev)
        self.ws_low = torch.tensor(WS_LOW, device=dev)
        self.ws_high = torch.tensor(WS_HIGH, device=dev)
        self.base_xy = torch.tensor(BASE_XY, device=dev)
        self.gen = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)
        self.forced_layouts = None

        self.cmd = torch.zeros(n, 3, device=dev)
        self.cmd_yaw = torch.zeros(n, device=dev)
        self.grip_target = torch.full((n,), 0.04, device=dev)
        self.hold = torch.zeros(n, dtype=torch.long, device=dev)
        self.reached = torch.zeros(n, len(tasks.STAGES[self.task]), dtype=torch.bool, device=dev)
        self.knocked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.done_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.succ_latch = torch.zeros(n, dtype=torch.bool, device=dev)   # success held at some tick of this step
        self.was_lifted = torch.zeros(n, dtype=torch.bool, device=dev)   # red cube picked up this episode (C1)
        self.rew_acc = torch.zeros(n, device=dev)
        self.ticks = torch.zeros(n, dtype=torch.long, device=dev)        # control ticks in this episode
        self._sub = 0
        self.tick_callback = None                 # f(env), called every control tick (videos)
        self.start_red = torch.zeros(n, 3, device=dev)
        self.layout = torch.zeros(n, 6, device=dev)
        self.tcp_hist = torch.zeros(n, 4, 3, device=dev)  # last 4 TCP positions, for jerk
        self.jerk_sum = torch.zeros(n, device=dev)
        self.jerk_cnt = torch.zeros(n, device=dev)
        self.peak_force = torch.zeros(n, device=dev)
        # filled when an episode ends; read by the eval harness
        self.last_episode = {k: torch.zeros(n, device=dev) for k in ["success", "time", "jerk", "peak_force", "lifted"]}
        self.last_reached = torch.zeros_like(self.reached)
        self.last_knocked = torch.zeros_like(self.knocked)
        self.episode_done = torch.zeros(n, dtype=torch.bool, device=dev)

    # ---------- scene ----------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.red = RigidObject(self.cfg.red)
        self.blue = RigidObject(self.cfg.blue)
        self.target = RigidObject(self.cfg.target)
        # 8 Oct: more objects (VLA_OBJECTS=multi, sim/envs/multi_env.py) are cfg attributes obj2, obj3, ...; none in the other envs
        self.extra = [RigidObject(getattr(self.cfg, f"obj{i}")) for i in range(2, 8) if hasattr(self.cfg, f"obj{i}")]
        self.contact = ContactSensor(self.cfg.contact)
        table = sim_utils.CuboidCfg(
            size=(1.2, 0.9, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.04, 0.04, 0.045) if tasks.CUBE_CYL else (0.45, 0.33, 0.22)),
        )
        # table frame x in [-0.75, 0.45]: the robot is mounted on the table, top at z = 0
        table.func("/World/envs/env_0/Table", table, translation=(TABLE_ORIGIN[0] - 0.15, 0.0, -0.375))
        ground = sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.27))
        ground.func("/World/ground", ground, translation=(0.0, 0.0, -0.75))
        light = sim_utils.DomeLightCfg(intensity=600.0, color=(0.9, 0.92, 1.0))
        light.func("/World/light", light)
        sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0, color=(1.0, 0.97, 0.92))
        sun.func("/World/sun", sun, orientation=_look_at((0, 0, 0), (-0.4, 0.5, -1.0)))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        for name in ["red", "blue", "target"]:
            self.scene.rigid_objects[name] = getattr(self, name)
        for i, o in enumerate(self.extra):
            self.scene.rigid_objects[f"obj{i + 2}"] = o
        self.scene.sensors["contact"] = self.contact
        if self.cfg.cameras:
            for name in ["front_cam", "wrist_cam"]:
                cam_cfg = getattr(self.cfg, name).replace(width=self.cfg.image_size, height=self.cfg.image_size)
                if name == "front_cam" and self.cfg.front_view == "near":
                    cam_cfg = cam_cfg.replace(offset=TiledCameraCfg.OffsetCfg(
                        pos=NEAR_FRONT[0], rot=_look_at(*NEAR_FRONT), convention="opengl"))
                self.scene.sensors[name] = TiledCamera(cam_cfg)

    # ---------- state in the table frame ----------
    def hand_rot(self):
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.hand_idx])

    def tcp_w(self):
        r = self.hand_rot()
        return self.robot.data.body_pos_w[:, self.hand_idx] + r[:, :, 2] * TCP_OFFSET

    def to_table(self, p_w):
        return p_w - self.table_origin

    def tcp_yaw(self):
        r = self.hand_rot()
        return torch.atan2(r[:, 1, 0], r[:, 0, 0])

    def grip_gap(self):
        return self.robot.data.joint_pos[:, self.finger_ids].sum(-1)

    def state(self):
        s = {
            "tcp": self.to_table(self.tcp_w()),
            "red": self.to_table(self.red.data.root_pos_w),
            "blue": self.to_table(self.blue.data.root_pos_w),
            "target": self.to_table(self.target.data.root_pos_w),
            "grip": self.grip_gap(),
            "red_speed": self.red.data.root_lin_vel_w.norm(dim=-1),
            "was_lifted": self.was_lifted,
            "cmd": self.cmd,
        }
        s["target"][:, 2] = 0.0
        return s

    # ---------- action ----------
    def _command(self, pos, yaw, opening):
        """Set the TCP command (table frame), its yaw and the finger opening in [0, 1]."""
        pos = pos.clamp(self.ws_low, self.ws_high)
        # keep the command inside the measured vertical-gripper reach (sim/probe_reach.py)
        rel = pos[:, :2] - self.base_xy
        r = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pos[:, :2] = self.base_xy + rel * r.clamp(*REACH) / r
        tcp = self.to_table(self.tcp_w())
        self.cmd = tcp + (pos - tcp).clamp(-MAX_LEAD, MAX_LEAD)
        self.cmd_yaw = yaw
        self.grip_target = 0.04 * opening

    def _pre_physics_step(self, actions):
        self.extras["log"] = {}
        self._sub = 0
        self.succ_latch[:] = False
        self.rew_acc[:] = 0.0
        a = actions.clamp(-1.0, 1.0)
        self._command(self.cmd + a[:, :3] * MAX_DXYZ, self.cmd_yaw + a[:, 3] * MAX_DYAW, (a[:, 4] + 1) / 2)

    def _block_start(self, block):
        """Called at the start of every control tick within an env step (the vocabulary env sets waypoints here)."""

    def _apply_action(self):
        if self._sub % CTRL_SUBSTEPS == 0:
            if self._sub > 0:
                self._tick()
            self._block_start(self._sub // CTRL_SUBSTEPS)
        self._sub += 1
        self._ik_step()

    def _ik_step(self):
        """One damped-least-squares IK step toward the commanded pose."""
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_idx, :, :7].clone()
        r = self.hand_rot()
        tcp_w = self.tcp_w()
        lever = tcp_w - self.robot.data.body_pos_w[:, self.hand_idx]
        lx, ly, lz = lever.unbind(-1)
        zero = torch.zeros_like(lx)
        skew = torch.stack([torch.stack([zero, -lz, ly], -1), torch.stack([lz, zero, -lx], -1),
                            torch.stack([-ly, lx, zero], -1)], dim=1)
        jac[:, :3] -= skew @ jac[:, 3:]                  # hand-frame Jacobian -> TCP
        e_p = self.cmd + self.table_origin - tcp_w
        c, s_ = torch.cos(self.cmd_yaw), torch.sin(self.cmd_yaw)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        # tool z down, tool x along yaw: rot_z(yaw) @ diag(1, -1, -1), as columns
        r_des = torch.stack([torch.stack([c, s_, z], -1), torch.stack([s_, -c, z], -1),
                             torch.stack([z, z, -o], -1)], dim=-1)
        e_r = 0.5 * torch.cross(r, r_des, dim=1).sum(-1)
        err = torch.cat([2.0 * e_p, ROT_WEIGHT * e_r], dim=-1).unsqueeze(-1)
        jjt = jac @ jac.transpose(1, 2) + 0.05**2 * torch.eye(6, device=self.device)
        dq = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err)).squeeze(-1).clamp(-0.04, 0.04)
        q = self.robot.data.joint_pos[:, self.arm_ids]
        self.robot.set_joint_position_target((q + dq).clamp(self.q_low, self.q_high), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.grip_target[:, None].repeat(1, 2), joint_ids=self.finger_ids)

    # ---------- task logic (tasks.py) ----------
    def _tick(self):
        """10 Hz bookkeeping: success hold, stages, knocked, metrics, reward."""
        s = self.state()
        self.ticks += 1
        self.was_lifted |= s["red"][:, 2] > tasks.LIFTED
        s["was_lifted"] = self.was_lifted
        self.ok = tasks.success(self.task, s)
        self.hold = torch.where(self.ok, self.hold + 1, torch.zeros_like(self.hold))
        self.done_ok = self.hold >= tasks.SUCCESS_HOLD
        self.succ_latch |= self.done_ok
        tasks.update_stages(self.task, s, self.reached, self.start_red, self.ok)
        red = s["red"]
        for cube in ([red, s["blue"]] if self.task in ("c2", "c3") else [red]):
            self.knocked |= (cube[:, 2] < -0.05) | (cube[:, :2].abs() > 0.5).any(-1)
        self._metrics(s)
        self.rew_acc += tasks.reward(self.task, s, self.done_ok)
        if self.tick_callback is not None:
            self.tick_callback(self)

    def _metrics(self, s):
        """Jerk and peak contact force, every control tick."""
        self.tcp_hist = torch.cat([self.tcp_hist[:, 1:], s["tcp"][:, None]], dim=1)
        step = self.ticks
        d3 = self.tcp_hist[:, 3] - 3 * self.tcp_hist[:, 2] + 3 * self.tcp_hist[:, 1] - self.tcp_hist[:, 0]
        valid = (step >= 4).float()
        self.jerk_sum += valid * d3.abs().sum(-1)
        self.jerk_cnt += valid * 3
        f = self.contact.data.net_forces_w_history.norm(dim=-1).amax(dim=1)   # (n, 2 cubes)
        self.peak_force = torch.maximum(self.peak_force, f.amax(-1))

    def _get_dones(self):
        self._tick()                             # the last tick of this env step
        terminated = (self.succ_latch & self.cfg.terminate_on_success) | self.knocked
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self):
        return self.rew_acc.clone()

    def _get_observations(self):
        s = self.state()
        yaw = self.tcp_yaw()
        goal = tasks.goal(self.task, s)
        obs = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08,
                         s["red"], s["blue"], s["target"], s["red"] - s["tcp"], goal - s["red"]], dim=-1)
        out = {"policy": obs}
        if self.cfg.cameras:
            out.update(self.images())
        return out

    def images(self):
        return {"front": self.scene.sensors["front_cam"].data.output["rgb"],
                "wrist": self.scene.sensors["wrist_cam"].data.output["rgb"]}

    # ---------- reset ----------
    def set_layouts(self, layouts):
        """Force the (num_envs, 6) layouts used by the next reset of each env (eval states)."""
        self.forced_layouts = layouts.to(self.device)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ran = self.episode_length_buf[env_ids] > 0
        done_ids = env_ids[ran]
        dt = self.cfg.sim.dt * CTRL_SUBSTEPS        # control tick
        self.last_episode["success"][done_ids] = (self.done_ok | self.succ_latch)[done_ids].float()
        self.last_episode["time"][done_ids] = self.ticks[done_ids].float() * dt
        self.last_episode["jerk"][done_ids] = self.jerk_sum[done_ids] / self.jerk_cnt[done_ids].clamp(min=1) / dt**3
        self.last_episode["peak_force"][done_ids] = self.peak_force[done_ids]
        self.last_episode["lifted"][done_ids] = self.was_lifted[done_ids].float()
        self.last_reached[done_ids] = self.reached[done_ids]
        self.last_knocked[done_ids] = self.knocked[done_ids]
        self.episode_done[done_ids] = True
        if len(done_ids) and "log" in self.extras:
            log = self.extras["log"]
            log["Episode/success"] = self.done_ok[done_ids].float()
            log["Episode/knocked"] = self.knocked[done_ids].float()
            for j, name in enumerate(tasks.STAGES[self.task]):
                log[f"Stage/{name}"] = self.reached[done_ids, j].float()
        super()._reset_idx(env_ids)

        k = len(env_ids)
        q = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)

        if self.forced_layouts is not None:
            lay = self.forced_layouts[env_ids]
        else:
            lay = tasks.sample_layouts(self.task, k, self.gen).to(self.device)
        self.layout[env_ids] = lay
        origin = self.table_origin[env_ids]
        unit_quat = torch.tensor([1.0, 0, 0, 0], device=self.device).repeat(k, 1)
        for obj, xy, z in [(self.red, lay[:, 0:2], tasks.HALF_Z[0]), (self.blue, lay[:, 2:4], tasks.HALF_Z[1]),
                           (self.target, lay[:, 4:6], 0.001)]:
            pos = origin + torch.cat([xy, torch.full((k, 1), z, device=self.device)], dim=-1)
            obj.write_root_pose_to_sim(torch.cat([pos, unit_quat], dim=-1), env_ids=env_ids)
            obj.write_root_velocity_to_sim(torch.zeros(k, 6, device=self.device), env_ids=env_ids)

        self.cmd[env_ids] = torch.tensor(HOME_TCP, device=self.device)
        self.cmd_yaw[env_ids] = 0.0
        self.grip_target[env_ids] = 0.04
        self.hold[env_ids] = 0
        self.reached[env_ids] = False
        self.knocked[env_ids] = False
        self.ok[env_ids] = False
        self.done_ok[env_ids] = False
        self.succ_latch[env_ids] = False
        self.was_lifted[env_ids] = False
        self.ticks[env_ids] = 0
        self.start_red[env_ids] = torch.cat([lay[:, 0:2], torch.full((k, 1), CUBE_HALF, device=self.device)], -1)
        self.tcp_hist[env_ids] = 0.0
        self.jerk_sum[env_ids] = 0.0
        self.jerk_cnt[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0
