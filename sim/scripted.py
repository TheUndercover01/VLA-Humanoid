"""Waypoint expert with privileged state. Used as the harness oracle and Oracle data source."""
import numpy as np

from sim.env import MAX_DXYZ, PandaTaskEnv
from sim.scene import CUBE_HALF

HOVER = 0.14
GRASP_Z = 0.026
CARRY_Z = 0.16


class Phase:
    def __init__(self, pos, grip, tol=0.012, wait=0, skill="reach", yaw=0.0):
        self.pos, self.grip, self.tol, self.wait, self.skill, self.yaw = np.asarray(pos, float), grip, tol, wait, skill, yaw


def plan_for(env: PandaTaskEnv):
    """Build the phase list from the current privileged state."""
    task = env.task
    red, goal = env.red(), env.goal()
    tcp = env.robot.tcp_pos()
    P = []

    def top(x, y, z):
        return np.array([x, y, z])

    def pick(red_xy):
        P.append(Phase(top(*red_xy, HOVER), 1.0, 0.02, skill="reach"))
        P.append(Phase(top(*red_xy, GRASP_Z), 1.0, 0.008, skill="reach"))
        P.append(Phase(top(*red_xy, GRASP_Z), 0.0, 0.005, wait=6, skill="pick-lift"))
        P.append(Phase(top(*red_xy, CARRY_Z), 0.0, 0.015, skill="pick-lift"))

    def place(goal_xyz):
        z_place = goal_xyz[2] + 0.004
        P.append(Phase(top(goal_xyz[0], goal_xyz[1], CARRY_Z), 0.0, 0.015, skill="push"))
        P.append(Phase(top(goal_xyz[0], goal_xyz[1], z_place), 0.0, 0.008, skill="place-down"))
        P.append(Phase(top(goal_xyz[0], goal_xyz[1], z_place), 1.0, 0.005, wait=6, skill="place-down"))
        P.append(Phase(top(goal_xyz[0], goal_xyz[1], CARRY_Z), 1.0, 0.02, skill="reach"))

    if task == "lift":
        pick(red[:2])
    elif task == "push":
        d = goal[:2] - red[:2]
        u = d / (np.linalg.norm(d) + 1e-9)
        behind = red[:2] - u * 0.075
        P.append(Phase(top(*behind, HOVER), 0.0, 0.02, skill="reach"))
        P.append(Phase(top(*behind, 0.03), 0.0, 0.01, skill="reach"))
        P.append(Phase(top(*(goal[:2] - u * 0.03), 0.03), 0.0, 0.01, skill="push"))
    else:
        if task == "c3":
            behind = red[:2] + np.array([0.075, 0.0])
            P.append(Phase(top(*behind, HOVER), 0.0, 0.02, skill="reach"))
            P.append(Phase(top(*behind, 0.03), 0.0, 0.01, skill="reach"))
            P.append(Phase(top(0.26 + 0.075, behind[1] * 0 + red[1], 0.03), 0.0, 0.01, skill="push"))
            P.append(Phase(top(0.26 + 0.075, red[1], HOVER), 1.0, 0.02, skill="reach"))
            pick_xy = np.array([0.26, red[1]])
        else:
            pick_xy = red[:2]
        pick(pick_xy)
        place(goal)
    return P


def run_expert(env: PandaTaskEnv, obs_cb=None):
    """Run one episode with the scripted expert. Returns the final info dict."""
    plan = plan_for(env)
    i, waited, info = 0, 0, {}
    while True:
        ph = plan[min(i, len(plan) - 1)]
        err = ph.pos - env.cmd
        step_vec = np.clip(err / MAX_DXYZ, -1, 1)
        a = np.array([*step_vec, 0.0, ph.grip * 2 - 1])
        if obs_cb:
            obs_cb(env, a, ph.skill)
        obs, r, term, trunc, info = env.step(a)
        if np.linalg.norm(ph.pos - env.robot.tcp_pos()) < ph.tol and i < len(plan):
            waited += 1
            if waited > ph.wait:
                i, waited = i + 1, 0
        if term or trunc:
            return info
        if i >= len(plan):                      # plan finished: hold still and let success register
            plan.append(Phase(plan[-1].pos, plan[-1].grip, 0.02, wait=30, skill=plan[-1].skill))
