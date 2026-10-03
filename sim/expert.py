"""Scripted waypoint expert with privileged state, batched over envs.

Test oracle for the eval harness and the source of the Oracle data. Each env gets a
list of phases (waypoint, yaw, grip, speed, tolerance, wait, skill) built from the start
state; a phase ends once the TCP is within tolerance for `wait` extra steps. Waypoints can
follow a cube's live pose, and pushes are closed-loop on the pushed cube.
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
ABS, REL_BLUE, REL_RED = 0, 1, 2  # waypoint xy is absolute, or an offset from the blue / red cube's live position


def yaw_of(q):
    """Yaw (rad) of (n, 4) quaternions (w, x, y, z)."""
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


class ScriptedExpert:
    def reset(self, env):
        """Plan every env from the current (just reset) state."""
        s = env.state()
        plans = [self._plan(env.task, s, i) for i in range(env.num_envs)]
        n, p = env.num_envs, max(len(pl) for pl in plans)
        dev = env.device
        self.wp = torch.zeros(n, p, 3, device=dev)
        self.yaw = torch.zeros(n, p, device=dev)
        self.speed = torch.zeros(n, p, device=dev)
        self.grip = torch.zeros(n, p, device=dev)
        self.tol = torch.zeros(n, p, device=dev)
        self.wait = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.rel = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.push_obj = torch.zeros(n, p, dtype=torch.long, device=dev)   # 0 none, 1 red, 2 blue
        self.skill = torch.zeros(n, p, dtype=torch.long, device=dev)
        self.count = torch.tensor([len(pl) for pl in plans], device=dev)
        for i, pl in enumerate(plans):
            for j, (pos, yaw, grip, speed, tol, wait, skill, rel, push_obj) in enumerate(pl):
                self.wp[i, j] = torch.tensor(pos, device=dev)
                self.yaw[i, j], self.grip[i, j], self.speed[i, j] = yaw, grip, speed
                self.tol[i, j], self.wait[i, j] = tol, wait
                self.skill[i, j], self.rel[i, j], self.push_obj[i, j] = SKILLS.index(skill), rel, push_obj
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.waited = torch.zeros(n, dtype=torch.long, device=dev)

    @staticmethod
    def _plan(task, s, i):
        red, blue, tgt = (s[k][i].tolist() for k in ("red", "blue", "target"))
        plan = []

        def add(pos, grip, tol, wait=0, skill="reach", rel=ABS, yaw=0.0, speed=1.0, push_obj=0):
            plan.append((pos, yaw, grip, speed, tol, wait, skill, rel, push_obj))

        def pick():
            # follows the red cube's live pose and yaw: an earlier push may have nudged or spun it
            add([0, 0, HOVER], OPEN, 0.02, rel=REL_RED)
            add([0, 0, GRASP_Z], OPEN, 0.008, rel=REL_RED)
            add([0, 0, GRASP_Z], CLOSED, 0.02, wait=6, skill="pick-lift", rel=REL_RED)   # timed: fingers close
            add([0, 0, CARRY_Z], CLOSED, 0.015, skill="pick-lift", rel=REL_RED)

        def place(x, y, z, rel=ABS):
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
            pick()
        elif task == "push":
            push(red, tgt, 1)
        elif task == "c1":
            pick()
            place(tgt[0], tgt[1], CUBE_HALF)
        elif task == "c2":
            pick()
            place(0.0, 0.0, 3 * CUBE_HALF, rel=REL_BLUE)
        elif task == "c3":
            push(blue, tgt, 2)
            pick()
            place(0.0, 0.0, 3 * CUBE_HALF, rel=REL_BLUE)
        return plan

    def act(self, env):
        """(n, 5) actions and the (n,) skill index of the current phase."""
        n = env.num_envs
        idx = torch.arange(n, device=env.device)
        ph = self.phase.clamp(max=self.count - 1)
        wp = self.wp[idx, ph].clone()
        s = env.state()
        blue_xy = s["blue"][:, :2]
        rel = self.rel[idx, ph]
        wp[rel == REL_BLUE, :2] += blue_xy[rel == REL_BLUE]
        wp[rel == REL_RED, :2] += s["red"][rel == REL_RED, :2]
        yaw = self.yaw[idx, ph].clone()
        # grasp yaw: the red cube's yaw mod 90 degrees, kept in [-45, 45] so the wrist (joint 7) can reach it
        grasp_yaw = torch.remainder(yaw_of(env.red.data.root_quat_w) + torch.pi / 4, torch.pi / 2) - torch.pi / 4
        yaw[rel == REL_RED] = grasp_yaw[rel == REL_RED]
        # closed-loop push: stay behind the cube on its current line to the target
        po = self.push_obj[idx, ph]
        pushing = po > 0
        if pushing.any():
            obj = torch.where((po == 1)[:, None], s["red"][:, :2], blue_xy)
            d = s["target"][:, :2] - obj
            dist = d.norm(dim=-1, keepdim=True)
            u = d / dist.clamp(min=1e-6)
            push_wp = obj + u * (dist.clamp(max=0.04) - 0.035)
            # if the cube has drifted off the line, do not cut across it: lift, go round to the new
            # spot behind it, come down and push again
            rel_tcp = s["tcp"][:, :2] - obj
            lateral = (u[:, 0] * rel_tcp[:, 1] - u[:, 1] * rel_tcp[:, 0]).abs()
            behind = -(rel_tcp * u).sum(-1) > 0.02
            # only for C3's long pushes of blue; the short push task does better without it
            off = ((lateral > 0.015) | ~behind) & (dist.squeeze(-1) > 0.10) & (po == 2)
            start = obj - u * 0.075
            above = (s["tcp"][:, :2] - start).norm(dim=-1) < 0.015
            push_wp = torch.where(off[:, None], start, push_wp)
            wp[pushing, :2] = push_wp[pushing]
            wp[pushing & off & ~above, 2] = HOVER
            # the gripper is symmetric under a half turn: take the push yaw closest to the current one,
            # so pushes along +-y do not flip the hand by 180 degrees mid-push
            line_yaw = torch.atan2(u[:, 1], u[:, 0])
            line_yaw = env.cmd_yaw + torch.remainder(line_yaw - env.cmd_yaw + torch.pi / 2, torch.pi) - torch.pi / 2
            # stay inside the wrist's range (yaw 0 is joint 7 = 0.785 rad, limits +-2.9 rad)
            line_yaw = torch.where(line_yaw > 2.0, line_yaw - torch.pi, line_yaw)
            line_yaw = torch.where(line_yaw < -2.0, line_yaw + torch.pi, line_yaw)
            yaw[pushing] = line_yaw[pushing]
            self._obj_dist = dist.squeeze(-1)
        a = torch.zeros(n, 5, device=env.device)
        speed = self.speed[idx, ph, None]
        a[:, :3] = ((wp - env.cmd) / MAX_DXYZ).clamp(-1, 1) * speed
        a[:, 3] = ((yaw - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)
        a[:, 4] = self.grip[idx, ph]
        self._wp, self._ph = wp, ph
        return a, self.skill[idx, ph]

    def update(self, env):
        """Advance phases whose waypoint was reached (checked on the TCP after the step)."""
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
