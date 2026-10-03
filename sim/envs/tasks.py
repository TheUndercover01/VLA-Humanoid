"""Task definitions shared by every policy: layouts, success, failure stages, reward.

Everything here is batched torch in the table frame: origin at the table centre,
x away from the robot, y left, z up, table top at z = 0. Ported from the MuJoCo
prototype (sim/env.py). Tasks differ only in what is defined in this file.
"""
import torch

TASKS = ["push", "lift", "c1", "c2", "c3"]
CUBE_HALF = 0.025
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
BASE_XY = (-0.5, 0.0)                        # robot base in the table frame
PUSH_REACH = (0.36, 0.74)                    # the pusher's start point must be this far from the base
SUCCESS_HOLD = 5                         # control steps (0.5 s)


def sample_layouts(task, n, gen):
    """(n, 6) layouts [red_xy, blue_xy, target_xy], rejection-sampled on the CPU."""
    k = 64
    lo, hi = torch.tensor(REGION[0]), torch.tensor(REGION[1])

    def uni(lo, hi):
        return lo + (hi - lo) * torch.rand(n, k, 2, generator=gen)

    red, blue, tgt = uni(lo, hi), uni(lo, hi), uni(lo, hi)
    if task == "push":                  # push target further out than the cube
        red = uni(torch.tensor(PUSH_RED_REGION[0]), torch.tensor(PUSH_RED_REGION[1]))
        tgt = red + uni(torch.tensor([0.08, -0.08]), torch.tensor([0.15, 0.08]))

    def far(a, b):
        return (a - b).norm(dim=-1) > 0.12

    box_lo, box_hi = torch.tensor(TARGET_BOX[0]), torch.tensor(TARGET_BOX[1])
    ok = far(red, blue) & far(red, tgt) & ((tgt > box_lo) & (tgt < box_hi)).all(-1)
    if task != "push":
        ok &= far(blue, tgt)
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
        return s["blue"] + torch.tensor([0.0, 0.0, 2 * CUBE_HALF], device=s["blue"].device)
    g = s["target"].clone()
    g[:, 2] = CUBE_HALF
    return g


def success(task, s):
    """Instantaneous success; the env requires it to hold for SUCCESS_HOLD steps.

    s: dict of (n, 3) tcp, red, blue, target, cmd and (n,) grip (finger gap, m), red_speed.
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
    return (d_xy < ON_TARGET) & (red[:, 2] < 0.04) & rest & open_ok


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


def reward(task, s, ok):
    """Dense shaped reward (reach -> grasp -> lift -> transport -> place) plus a bonus for every
    step in the success state. A per-step bonus (rather than a one-off bonus with termination,
    as in the prototype) keeps finishing worth more than hovering near the goal."""
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
        # Grasp is detected by contact geometry, not height, and carrying is rewarded by the 3-D
        # distance to the resting pose at the goal, so lowering the cube is never penalised.
        # "At goal" pays whether or not the cube is still held, so letting go there is a gain.
        # (The prototype's height-based "held" made the policy hover over the goal.)
        grasped = ((tcp - red).norm(dim=-1) < 0.03) & (s["grip"] > 0.03) & (s["grip"] < RELEASED)
        if task == "c3":
            grasped = grasped & (on > 0)
        grasped = grasped.float()
        d3 = (red - g).norm(dim=-1)
        rew = rew + 1.0 * grasped + 3.0 * grasped * (1 - torch.tanh(5 * d3))
        at_goal = ((d_goal < 0.02) & ((red[:, 2] - g[:, 2]).abs() < 0.01)).float()
        if task == "c3":
            at_goal = at_goal * on
        rew = rew + at_goal * (3.0 + 2.0 * s["grip"] / 0.08)
    rew = rew - 0.01 * ((s["cmd"] - tcp) ** 2).sum(-1)
    return rew + SUCCESS_REWARD * ok.float()
