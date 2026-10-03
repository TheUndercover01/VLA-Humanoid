"""Task logic (sim/envs/tasks.py) on hand-built states: success checks, stages, layouts, reward ordering.

    python -m pytest tests -q
"""
import torch

from sim.envs import tasks
from sim.envs.tasks import CUBE_HALF, RELEASED


def state(red, blue=(0.1, -0.15, CUBE_HALF), target=(0.0, 0.0, 0.0), tcp=(0.0, 0.0, 0.25), grip=0.08,
          speed=0.0, cmd=None, lifted=True):
    t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
    tcp = t(tcp)
    return {"red": t(red), "blue": t(blue), "target": t(target), "tcp": tcp, "grip": t(grip),
            "red_speed": t(speed), "cmd": tcp.clone() if cmd is None else t(cmd),
            "was_lifted": torch.tensor([lifted])}


def test_c1_success_needs_cube_on_target_resting_and_released():
    on = (0.01, 0.0, CUBE_HALF)
    assert tasks.success("c1", state(on))
    assert not tasks.success("c1", state((0.05, 0.0, CUBE_HALF)))          # 5 cm off the target
    assert not tasks.success("c1", state(on, speed=0.1))                     # still moving
    assert not tasks.success("c1", state(on, grip=0.05))                     # fingers still on the 5 cm cube
    assert not tasks.success("c1", state((0.0, 0.0, 0.08)))                  # held above the target
    assert not tasks.success("c1", state(on, lifted=False))                  # pushed there, never picked up


def test_c2_success_needs_red_on_blue():
    blue = (0.1, -0.1, CUBE_HALF)
    on_blue = (0.1, -0.1, 3 * CUBE_HALF)
    assert tasks.success("c2", state(on_blue, blue=blue))
    assert not tasks.success("c2", state((0.1, -0.06, 3 * CUBE_HALF), blue=blue))   # 4 cm off-centre
    assert not tasks.success("c2", state((0.25, 0.1, CUBE_HALF), blue=blue))        # beside it on the table


def test_c3_needs_blue_on_target_too():
    on_blue = lambda b: (b[0], b[1], 3 * CUBE_HALF)  # noqa: E731
    blue_on_tgt = (0.0, 0.0, CUBE_HALF)
    blue_off = (0.15, 0.0, CUBE_HALF)
    assert tasks.success("c3", state(on_blue(blue_on_tgt), blue=blue_on_tgt))
    assert not tasks.success("c3", state(on_blue(blue_off), blue=blue_off))


def test_lift_and_push():
    assert tasks.success("lift", state((0.0, 0.0, 0.12)))
    assert not tasks.success("lift", state((0.0, 0.0, CUBE_HALF)))
    assert tasks.success("push", state((0.01, 0.01, CUBE_HALF)))
    assert not tasks.success("push", state((0.0, 0.0, 0.10)))                # carried, not pushed


def test_failure_stage_is_first_unreached_stage():
    names = tasks.STAGES["c1"]
    reached = torch.tensor([[True, True, False, False, False]])
    assert tasks.failure_stage("c1", reached, torch.tensor([False])) == [names[2]]
    assert tasks.failure_stage("c1", reached, torch.tensor([True])) == ["knocked"]
    assert tasks.failure_stage("c1", torch.ones(1, 5, dtype=torch.bool), torch.tensor([False])) == ["place"]


def test_update_stages_marks_grasp_only_when_lifted_in_the_hand():
    task = "c1"
    reached = torch.zeros(1, len(tasks.STAGES[task]), dtype=torch.bool)
    start = torch.tensor([[0.1, 0.0, CUBE_HALF]])
    s = state((0.1, 0.0, CUBE_HALF), tcp=(0.1, 0.0, CUBE_HALF), grip=0.05)
    tasks.update_stages(task, s, reached, start, tasks.success(task, s))
    assert reached[0, 0] and not reached[0, 1]                               # reached, not yet grasped
    s = state((0.1, 0.0, 0.10), tcp=(0.1, 0.0, 0.10), grip=0.05)
    tasks.update_stages(task, s, reached, start, tasks.success(task, s))
    assert reached[0, 1] and reached[0, 2]                                   # grasped and lifted


def test_layouts_are_reproducible_and_inside_the_workspace():
    for task in tasks.TASKS:
        a = tasks.sample_layouts(task, 200, torch.Generator().manual_seed(5))
        b = tasks.sample_layouts(task, 200, torch.Generator().manual_seed(5))
        assert torch.equal(a, b)
        red, blue, tgt = a[:, 0:2], a[:, 2:4], a[:, 4:6]
        assert ((red - blue).norm(dim=-1) > 0.12).all()
        box_lo, box_hi = torch.tensor(tasks.TARGET_BOX[0]), torch.tensor(tasks.TARGET_BOX[1])
        assert ((tgt > box_lo) & (tgt < box_hi)).all()
        base = torch.tensor(tasks.BASE_XY)
        assert ((red - base).norm(dim=-1) < 0.77).all()                    # every cube is reachable


def test_c3_push_start_is_reachable():
    lay = tasks.sample_layouts("c3", 500, torch.Generator().manual_seed(0))
    blue, tgt = lay[:, 2:4], lay[:, 4:6]
    u = (tgt - blue) / (tgt - blue).norm(dim=-1, keepdim=True)
    r = (blue - 0.075 * u - torch.tensor(tasks.BASE_XY)).norm(dim=-1)
    assert ((r > tasks.PUSH_REACH[0]) & (r < tasks.PUSH_REACH[1])).all()


def test_reward_prefers_released_at_goal_over_holding_there():
    """The bugs that made PPO hover over the goal or sit on the release threshold."""
    for task, goal_red in [("c1", (0.0, 0.0, CUBE_HALF)), ("c2", (0.1, -0.15, 3 * CUBE_HALF))]:
        held = state(goal_red, tcp=goal_red, grip=0.05)
        at_threshold = state(goal_red, tcp=goal_red, grip=RELEASED - 0.001)
        released = state(goal_red, tcp=(goal_red[0], goal_red[1], 0.16), grip=0.08)
        no, yes = torch.tensor([False]), torch.tensor([True])
        r_held = tasks.reward(task, held, no)
        r_thr = tasks.reward(task, at_threshold, no)
        r_rel = tasks.reward(task, released, yes)
        assert r_held < r_thr < r_rel, (task, r_held, r_thr, r_rel)


def test_reward_does_not_drop_when_lowering_or_releasing_near_goal():
    """The bug that made the vocabulary agent never put the cube down."""
    carried_high = state((0.0, 0.0, 0.16), tcp=(0.0, 0.0, 0.16), grip=0.05)
    dropped_near = state((0.02, 0.0, CUBE_HALF), tcp=(0.0, 0.0, 0.16), grip=0.08)
    no = torch.tensor([False])
    assert tasks.reward("c1", dropped_near, no) > tasks.reward("c1", carried_high, no)


def test_c1_pushing_to_the_goal_earns_less_than_lifting():
    """The vocabulary agent solved C1 by pushing; goal credit now needs the cube picked up first."""
    no = torch.tensor([False])
    pushed_to_goal = state((0.0, 0.0, CUBE_HALF), tcp=(-0.04, 0.0, 0.03), grip=0.03, lifted=False)
    lifted_far = state((0.15, 0.1, 0.10), tcp=(0.15, 0.1, 0.10), grip=0.05, lifted=True)
    assert tasks.reward("c1", lifted_far, no) > tasks.reward("c1", pushed_to_goal, no)
