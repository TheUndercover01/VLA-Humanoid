"""Scripted waypoint expert with privileged state, batched over envs.

Test oracle for the eval harness and the source of the Oracle data. Each env gets a
list of phases (waypoint, yaw, grip, speed, tolerance, wait, skill) built from the start
state; a phase ends once the TCP is within tolerance for `wait` extra steps.
"""
import math

import torch

from sim.envs.panda_env import MAX_DXYZ, MAX_DYAW
from sim.envs.tasks import CUBE_HALF

HOVER = 0.14
GRASP_Z = 0.026
CARRY_Z = 0.16
PUSH_Z = 0.03                   # lower and the fingertips catch the table (sweep on non-eval layouts)
PUSH_SPEED = 0.5                # fraction of MAX_DXYZ: push slowly so the cube does not slide on
SKILLS = ["reach", "push", "pick-lift", "place-down"]
OPEN, CLOSED = 1.0, -1.0
PUSH_GRIP = -0.25               # 3 cm finger gap: both fingertips touch the cube face, which cannot pass between


class ScriptedExpert:
    def reset(self, env):
        """Plan every env from the current (just reset) state."""
        s = env.state()
        plans = [self._plan(env.task, s, i) for i in range(env.num_envs)]
        n, p = env.num_envs, max(len(pl) for pl in plans)
        dev = env.device
        self.wp = torch.zeros(n, p, 3, device=dev)        # xyz, or xy offset from blue if rel_blue
        self.yaw = torch.zeros(n, p, device=dev)
        self.speed = torch.zeros(n, p, device=dev)
        self.grip = torch.zeros(n, p, device=dev)
        self.tol = torch.zeros(n, p, device=dev)
        self.wait = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.rel_blue = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.push_obj = torch.zeros(n, p, dtype=torch.long, device=dev)   # 0 none, 1 red, 2 blue
        self.skill = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.count = torch.tensor([len(pl) for pl in plans], device=dev)
        for i, pl in enumerate(plans):
            for j, (pos, yaw, grip, speed, tol, wait, skill, rel, push_obj) in enumerate(pl):
                self.push_obj[i, j] = push_obj
                self.wp[i, j] = torch.tensor(pos, device=dev)
                self.yaw[i, j], self.grip[i, j], self.speed[i, j] = yaw, grip, speed
                self.tol[i, j], self.wait[i, j] = tol, wait
                self.skill[i, j], self.rel_blue[i, j] = SKILLS.index(skill), rel
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.waited = torch.zeros(n, dtype=torch.long, device=dev)

    @staticmethod
    def _plan(task, s, i):
        red, blue, tgt = (s[k][i].tolist() for k in ("red", "blue", "target"))
        plan = []

        def add(pos, grip, tol, wait=0, skill="reach", rel=False, yaw=0.0, speed=1.0, push_obj=0):
            plan.append((pos, yaw, grip, speed, tol, wait, skill, rel, push_obj))

        def pick(x, y):
            add([x, y, HOVER], OPEN, 0.02)
            add([x, y, GRASP_Z], OPEN, 0.008)
            add([x, y, GRASP_Z], CLOSED, 0.02, wait=6, skill="pick-lift")      # timed: fingers close
            add([x, y, CARRY_Z], CLOSED, 0.015, skill="pick-lift")

        def place(x, y, z, rel=False):
            add([x, y, CARRY_Z], CLOSED, 0.015, skill="place-down", rel=rel)
            add([x, y, z + 0.004], CLOSED, 0.008, skill="place-down", rel=rel)
            add([x, y, z + 0.004], OPEN, 0.02, wait=6, skill="place-down", rel=rel)   # timed: fingers open
            add([x, y, CARRY_Z], OPEN, 0.02, rel=rel)

        def push(obj, to, which):
            d = torch.tensor(to[:2]) - torch.tensor(obj[:2])
            u = d / (d.norm() + 1e-9)
            behind = torch.tensor(obj[:2]) - u * 0.075
            end = torch.tensor(to[:2]) - u * 0.035
            # hand x along the push, so the two fingertips sit side by side on the cube face;
            # the gripper is symmetric under a half turn, so keep yaw in [-pi/2, pi/2]
            yaw = math.atan2(u[1], u[0])
            yaw = yaw - math.pi if yaw > math.pi / 2 else yaw + math.pi if yaw < -math.pi / 2 else yaw
            add([*behind.tolist(), HOVER], PUSH_GRIP, 0.02, yaw=yaw)
            add([*behind.tolist(), PUSH_Z], PUSH_GRIP, 0.01, yaw=yaw)
            add([*end.tolist(), PUSH_Z], PUSH_GRIP, 0.008, skill="push", yaw=yaw, speed=PUSH_SPEED, push_obj=which)
            add([*end.tolist(), HOVER], PUSH_GRIP, 0.02, yaw=yaw)

        if task == "lift":
            pick(red[0], red[1])
        elif task == "push":
            push(red, tgt, 1)
        elif task == "c1":
            pick(red[0], red[1])
            place(tgt[0], tgt[1], CUBE_HALF)
        elif task == "c2":
            pick(red[0], red[1])
            place(0.0, 0.0, 3 * CUBE_HALF, rel=True)
        elif task == "c3":
            push(blue, tgt, 2)
            pick(red[0], red[1])
            place(0.0, 0.0, 3 * CUBE_HALF, rel=True)
        return plan

    def act(self, env):
        """(n, 5) actions and the (n,) skill index of the current phase."""
        n = env.num_envs
        idx = torch.arange(n, device=env.device)
        ph = self.phase.clamp(max=self.count - 1)
        wp = self.wp[idx, ph].clone()
        s = env.state()
        blue_xy = s["blue"][:, :2]
        rel = self.rel_blue[idx, ph]
        wp[rel, :2] = wp[rel, :2] + blue_xy[rel]
        yaw = self.yaw[idx, ph].clone()
        # closed-loop push: stay behind the cube on its current line to the target
        po = self.push_obj[idx, ph]
        pushing = po > 0
        if pushing.any():
            obj = torch.where((po == 1)[:, None], s["red"][:, :2], blue_xy)
            d = s["target"][:, :2] - obj
            dist = d.norm(dim=-1, keepdim=True)
            u = d / dist.clamp(min=1e-6)
            push_wp = obj + u * (dist.clamp(max=0.04) - 0.035)
            wp[pushing, :2] = push_wp[pushing]
            # the gripper is symmetric under a half turn: take the push yaw closest to the current one,
            # so pushes along +-y do not flip the hand by 180 degrees mid-push
            line_yaw = torch.atan2(u[:, 1], u[:, 0])
            line_yaw = env.cmd_yaw + torch.remainder(line_yaw - env.cmd_yaw + torch.pi / 2, torch.pi) - torch.pi / 2
            yaw[pushing] = line_yaw[pushing]
            self._obj_dist = dist.squeeze(-1)
        a = torch.zeros(n, 5, device=env.device)
        speed = self.speed[idx, ph, None]
        a[:, :3] = ((wp - env.cmd) / MAX_DXYZ).clamp(-1, 1) * speed
        a[:, 3] = ((yaw - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)
        a[:, 4] = self.grip[idx, ph]
        # advance phases that are within tolerance (checked on the TCP after this step, as in the prototype)
        self._wp, self._ph = wp, ph
        return a, self.skill[idx, ph]

    def update(self, env):
        tcp = env.state()["tcp"]
        idx = torch.arange(env.num_envs, device=env.device)
        inside = (self._wp - tcp).norm(dim=-1) < self.tol[idx, self._ph]
        pushing = self.push_obj[idx, self._ph] > 0
        if pushing.any():                       # a push ends when the cube is on the target
            inside = torch.where(pushing, self._obj_dist < self.tol[idx, self._ph], inside)
        self.waited = torch.where(inside, self.waited + 1, self.waited)
        nxt = inside & (self.waited > self.wait[idx, self._ph]) & (self.phase < self.count)
        self.phase = torch.where(nxt, self.phase + 1, self.phase)
        self.waited = torch.where(nxt, torch.zeros_like(self.waited), self.waited)
