"""Task definitions shared by every policy: layouts, success, failure stages, reward.

Everything here is batched torch in the table frame: origin at the table centre,
x away from the robot, y left, z up, table top at z = 0. Ported from the MuJoCo
prototype (sim/env.py). Tasks differ only in what is defined in this file.
"""
import torch

TASKS = ["push", "lift", "c1", "c2", "c3"]
CUBE_HALF = 0.025
# 6 Oct (user: the real clips use a cube and a cylinder): VLA_OBJECTS=cube_cylinder makes object 0 (the "red" slot) a 5 cm cube and
# object 1 (the "blue" slot) a cylinder, radius 3 cm (the real can is 7.6 cm wide; the Panda opens 8 cm, so 6 cm leaves room to
# grasp), height 8 cm (the real one 8.5). Default: the two cubes of the earlier runs.
import os  # noqa: E402
OBJECTS = os.environ.get("VLA_OBJECTS", "red_blue")
CYL_R, CYL_H = 0.03, 0.08
CUBE_CYL = OBJECTS in ("cube_cylinder", "multi")        # objects 0 and 1 are a cube and a cylinder
HALF_Z = (CUBE_HALF, CYL_H / 2 if CUBE_CYL else CUBE_HALF)
# 8 Oct (user: "objects can be anything", many objects, same clips): VLA_OBJECTS=multi has six object types, slot k = type k (cube and
# cylinder as before). Cuboid: half extents (x, y, z); cylinder: (radius, half height). Every horizontal extent fits the Panda's 8 cm
# opening at any yaw (brick 6.5 x 4: 7.6 across the diagonal, tile 5.5 x 5.5: 7.8), so the hand never needs to turn to the object.
ROSTER = [
    ("cube", "box", (CUBE_HALF, CUBE_HALF, CUBE_HALF), (0.92, 0.92, 0.92)),
    ("cylinder", "cyl", (CYL_R, CYL_H / 2), (0.8, 0.1, 0.1)),
    ("brick", "box", (0.0325, 0.02, 0.02), (0.95, 0.8, 0.1)),
    ("disc", "cyl", (0.035, 0.0125), (0.1, 0.3, 0.9)),
    ("tile", "box", (0.0275, 0.0275, 0.0125), (0.6, 0.2, 0.8)),
    ("pillar", "box", (0.02, 0.02, 0.04), (0.1, 0.7, 0.3)),
]
N_TYPES = len(ROSTER)
ROSTER_HALF_Z = [r[2][2] if r[1] == "box" else r[2][1] for r in ROSTER]
# 9 Oct (user: "simplify: 3 objects in total, 1 of them new, and 4 when the student eval goes on"): VLA_TRAIN_TYPES="0,1,3" VLA_HOLDOUT_OBJECT=4 makes
# the VLA's data contain only the cube, the cylinder and the disc (3 on the table, always), the tile is the object it never saw (it is on the table
# at evaluation: 4 objects). Default: all types but the pillar, the pillar is held out (the first objects run).
HOLDOUT_OBJECT = int(os.environ.get("VLA_HOLDOUT_OBJECT", 5))   # in the teacher's training, never in the VLA's data
_TT = os.environ.get("VLA_TRAIN_TYPES", "")
TRAIN_TYPES = tuple(int(x) for x in _TT.split(",")) if _TT else tuple(t for t in range(N_TYPES) if t != HOLDOUT_OBJECT)
N_ON_TABLE = int(os.environ.get("VLA_N_OBJECTS", 0))            # 0: 3 or 4 at random; otherwise exactly this many in a training scene
NO_PUSH = (5,)                                          # the pillar topples
STACK_BASES = (0, 1, 3, 4)                              # cube, cylinder, disc, tile have a flat top wide enough to stack on
# (top, base) pairs the VLA never sees together (the teacher does): brick on disc, cylinder on tile, disc on cube
HOLDOUT_PAIRS = () if _TT else ((2, 3), (1, 4), (3, 0))
# 6 Oct night (user: "fix the target, it is easier to learn"): VLA_FIXED_TARGET="x,y" puts the target pad at one table-frame position in every
# layout and every episode (the real clips have one target marker), training and evaluation alike. Default: random, as before.
_FT = os.environ.get("VLA_FIXED_TARGET", "")
FIXED_TARGET = tuple(float(v) for v in _FT.split(",")) if _FT else None   # centre height of each object resting on the table
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
    "c3": "push the blue cube onto the green target, then stack the red cube on it",
}
# Measured with sim/probe_reach.py: a vertical gripper reaches up to ~0.78 m from the robot
# base (x = 0.275 at y = 0) and not closer than ~0.3 m. Everything is placed well inside that.
REGION = ((-0.12, -0.20), (0.18, 0.20))      # cubes and target are placed in this xy box
PUSH_RED_REGION = ((-0.12, -0.15), (0.08, 0.15))
TARGET_BOX = ((-0.15, -0.22), (0.24, 0.22))  # push targets must stay inside this box
ON_TARGET = 0.03                             # xy tolerance for "cube is on the target"
RELEASED = 0.065                             # finger gap (m) beyond the 5 cm cube: it has been let go
LIFTED = CUBE_HALF + 0.05                    # C1 must pick the cube up: its centre was at least 5 cm off the table
BASE_XY = (-0.5, 0.0)                        # robot base in the table frame
HOME_TCP = (0.0, 0.0, 0.25)                  # TCP at the robot's home joint pose (sim/probe_reach.py)
PUSH_REACH = (0.36, 0.68)                    # the pusher's start point must be this far from the base
EPISODE_S = {"push": 15.0, "lift": 15.0, "c1": 15.0, "c2": 15.0, "c3": 25.0}   # C3 chains push + pick + place
SUCCESS_HOLD = 5                         # control steps (0.5 s)


def sample_layouts(task, n, gen):
    """(n, 6) layouts [red_xy, blue_xy, target_xy], rejection-sampled on the CPU."""
    k = 64
    lo, hi = torch.tensor(REGION[0]), torch.tensor(REGION[1])

    def uni(lo, hi):
        return lo + (hi - lo) * torch.rand(n, k, 2, generator=gen)

    red, blue, tgt = uni(lo, hi), uni(lo, hi), uni(lo, hi)
    if FIXED_TARGET:
        tgt = torch.tensor(FIXED_TARGET, dtype=torch.float32).expand(n, k, 2).clone()
    if task == "push" and not FIXED_TARGET:   # push target further out than the cube
        red = uni(torch.tensor(PUSH_RED_REGION[0]), torch.tensor(PUSH_RED_REGION[1]))
        tgt = red + uni(torch.tensor([0.08, -0.08]), torch.tensor([0.15, 0.08]))

    def far(a, b):
        return (a - b).norm(dim=-1) > 0.12

    box_lo, box_hi = torch.tensor(TARGET_BOX[0]), torch.tensor(TARGET_BOX[1])
    ok = far(red, blue) & far(red, tgt) & ((tgt > box_lo) & (tgt < box_hi)).all(-1)
    if task != "push" or FIXED_TARGET:
        ok &= far(blue, tgt)
    if task == "push" and FIXED_TARGET:     # red is pushed: the gripper must get behind it
        u = (tgt - red) / (tgt - red).norm(dim=-1, keepdim=True)
        r = (red - 0.075 * u - torch.tensor(BASE_XY)).norm(dim=-1)
        ok &= (r > PUSH_REACH[0]) & (r < PUSH_REACH[1])
    if task == "c3":
        # the gripper must be able to get behind blue (7.5 cm back along the push line)
        u = (tgt - blue) / (tgt - blue).norm(dim=-1, keepdim=True)
        r = (blue - 0.075 * u - torch.tensor(BASE_XY)).norm(dim=-1)
        ok &= (r > PUSH_REACH[0]) & (r < PUSH_REACH[1])
    assert ok.any(1).all(), "layout rejection sampling failed"
    idx = ok.float().argmax(1)
    pick = lambda x: x[torch.arange(n), idx]  # noqa: E731
    return torch.cat([pick(red), pick(blue), pick(tgt)], dim=-1)


def goal(task, s):
    """Where the red cube's centre should end up."""
    if task in ("c2", "c3"):
        return s["blue"] + torch.tensor([0.0, 0.0, HALF_Z[0] + HALF_Z[1]], device=s["blue"].device)
    g = s["target"].clone()
    g[:, 2] = CUBE_HALF
    return g


def success(task, s):
    """Instantaneous success; the env requires it to hold for SUCCESS_HOLD steps.

    s: dict of (n, 3) tcp, red, blue, target, cmd and (n,) grip (finger gap, m), red_speed, and
    was_lifted (n,) bool: the red cube has been above LIFTED at some point this episode.
    """
    red, g = s["red"], goal(task, s)
    d_xy = (red[:, :2] - g[:, :2]).norm(dim=-1)
    if task == "c3":
        return blue_on_target(s) & success("c2", s)
    if task == "lift":
        return red[:, 2] > 0.10
    if task == "push":
        return (d_xy < 0.03) & (red[:, 2] < 0.04)
    rest = s["red_speed"] < 0.03
    open_ok = s["grip"] > RELEASED
    if task == "c2":
        return (d_xy < 0.025) & ((red[:, 2] - g[:, 2]).abs() < 0.012) & rest & open_ok
    # C1 says "pick up ... and place": pushing the cube onto the target does not count
    return (d_xy < ON_TARGET) & (red[:, 2] < 0.04) & rest & open_ok & s["was_lifted"]


def blue_on_target(s):
    d = (s["blue"][:, :2] - s["target"][:, :2]).norm(dim=-1)
    return (d < ON_TARGET) & (s["blue"][:, 2] < 0.04)


def update_stages(task, s, reached, start_red, ok):
    """Mark the stages reached this step. reached: (n, len(STAGES[task])) bool, in place."""
    names = STAGES[task]
    col = {name: i for i, name in enumerate(names)}
    red, g = s["red"], goal(task, s)
    near = (s["tcp"] - red).norm(dim=-1) < 0.06
    if task == "push":
        reached[:, col["push"]] |= (red[:, :2] - start_red[:, :2]).norm(dim=-1) > 0.05
    if task == "c3":
        reached[:, col["push"]] |= blue_on_target(s)
    reached[:, col["reach"]] |= near
    if "grasp" in col:
        lifted = red[:, 2] > CUBE_HALF + 0.03
        reached[:, col["grasp"]] |= reached[:, col["reach"]] & near & lifted & (s["grip"] < 0.07)
        reached[:, col["lift"]] |= reached[:, col["grasp"]] & (red[:, 2] > CUBE_HALF + 0.06)
    if "transport" in col:
        reached[:, col["transport"]] |= reached[:, col["lift"]] & ((red[:, :2] - g[:, :2]).norm(dim=-1) < 0.06)
    reached |= ok[:, None]


def failure_stage(task, reached, knocked):
    """Name of the stage where each episode failed (only meaningful for failures)."""
    names = STAGES[task]
    out = []
    for r, k in zip(reached.tolist(), knocked.tolist()):
        if k:
            out.append("knocked")
        else:
            out.append(next((names[i] for i, done in enumerate(r) if not done), "place"))
    return out


SUCCESS_REWARD = 5.0


def reward(task, s, held_ok):
    """Dense shaped reward (reach -> grasp -> lift -> transport -> place) plus a bonus for every
    step once success has held for SUCCESS_HOLD steps. A per-step bonus (rather than a one-off
    bonus with termination, as in the prototype) keeps finishing worth more than hovering, and
    requiring the hold stops a policy from flickering in and out of the success state."""
    red, tcp, g = s["red"], s["tcp"], goal(task, s)
    d_reach = (tcp - (red + torch.tensor([0.0, 0.0, 0.01], device=red.device))).norm(dim=-1)
    held = ((red[:, 2] > CUBE_HALF + 0.02) & (d_reach < 0.05) & (s["grip"] < 0.07)).float()
    d_goal = (red[:, :2] - g[:, :2]).norm(dim=-1)
    rew = 0.5 * (1 - torch.tanh(8 * d_reach))
    if task == "push":
        rew = 0.3 * (1 - torch.tanh(8 * d_reach)) + 1.5 * (1 - torch.tanh(6 * d_goal))
    elif task == "lift":
        rew = rew + 2.0 * held + 4.0 * (red[:, 2] - CUBE_HALF).clamp(0, 0.1) / 0.1 * held
    else:
        if task == "c3":
            # first get blue onto the target; the stacking terms only count once it is there
            blue, tgt = s["blue"], s["target"]
            d_blue = (blue[:, :2] - tgt[:, :2]).norm(dim=-1)
            on = blue_on_target(s).float()
            d_tcp_blue = (tcp - blue).norm(dim=-1)
            rew = (1 - on) * 0.5 * (1 - torch.tanh(8 * d_tcp_blue)) + 2.0 * (1 - torch.tanh(6 * d_blue))
            rew = rew + on * 0.5 * (1 - torch.tanh(8 * d_reach))
        # Grasp is detected by contact geometry, not height. The cube's 3-D distance to its resting
        # pose at the goal pays whether or not it is held, so lowering it and letting go near the
        # goal never lose much. At the goal pose the at-goal term grows as the gripper opens, and
        # the grasp and stay-near-the-cube terms switch off, so the hand lets go and leaves it at rest.
        # History of what went wrong (PROGRESS.md): a height-based "held" made PPO hover over the
        # goal; grasp and success terms paid side by side made it sit on the release threshold;
        # carry credit only while grasped made the vocabulary agent never put the cube down; the
        # reach term made the raw C2 agent keep jostling the stacked cube.
        at_goal = (d_goal < 0.02) & ((red[:, 2] - g[:, 2]).abs() < 0.01)
        grasped = ((tcp - red).norm(dim=-1) < 0.03) & (s["grip"] > 0.03) & (s["grip"] < RELEASED) & ~at_goal
        stage_on = torch.ones_like(d_goal)
        if task == "c1":
            # goal credit only once the cube has been picked up (otherwise RL pushes it there);
            # before that, reward lifting it while it is grasped
            lifted = s["was_lifted"].float()
            stage_on = lifted
            grasped_now = ((tcp - red).norm(dim=-1) < 0.03) & (s["grip"] > 0.03) & (s["grip"] < RELEASED)
            rew = rew + 2.0 * grasped_now.float() * (1 - lifted) * ((red[:, 2] - CUBE_HALF) / 0.05).clamp(0, 1)
        if task == "c3":
            stage_on = on
            grasped, at_goal = grasped & (on > 0), at_goal & (on > 0)
        if task == "c1":
            at_goal = at_goal & (stage_on > 0)
        grasped, at_goal = grasped.float(), at_goal.float()
        d3 = (red - g).norm(dim=-1)
        rew = rew - 0.5 * (1 - torch.tanh(8 * d_reach)) * at_goal * stage_on
        rew = rew + 1.0 * grasped + 3.0 * stage_on * (1 - torch.tanh(5 * d3))
        rew = rew + at_goal * (3.0 + 3.0 * s["grip"] / 0.08)
    rew = rew - 0.01 * ((s["cmd"] - tcp) ** 2).sum(-1)
    return rew + SUCCESS_REWARD * held_ok.float()
