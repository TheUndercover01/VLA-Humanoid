"""Clip -> reward reference (motion/skill_ref.py) on hand-built clips.

    python -m pytest tests -q
"""
import numpy as np

from motion import skill_ref
from sim.envs.tasks import CUBE_HALF


def place_clip(start=(0.2, 0.0), end=(0.0, 0.0), height=0.14, n=30, release=0.8):
    """Carry the cube from start at `height` to end, lower it and open."""
    t = np.linspace(0, 1, n)
    xy = np.array(start) + (np.array(end) - np.array(start)) * np.clip(t / 0.6, 0, 1)[:, None]
    z = CUBE_HALF + height * np.clip((0.8 - t) / 0.2, 0, 1)
    obj = np.column_stack([xy, z])
    ee = obj + [0, 0, 0.0]
    grip = (t < release).astype(float)
    return {"t": t * 3, "ee_pos": ee, "ee_yaw": np.zeros(n), "grip": grip, "obj_pos": obj, "skill": "place-down"}


def lift_clip(at=(0.0, 0.1), height=0.12, n=20):
    t = np.linspace(0, 1, n)
    obj = np.column_stack([np.full(n, at[0]), np.full(n, at[1]), CUBE_HALF + height * t])
    return {"t": t * 2, "ee_pos": obj.copy(), "ee_yaw": np.zeros(n), "grip": np.ones(n), "obj_pos": obj,
            "skill": "pick-lift"}


def moved(c, shift, yaw):
    """The same clip filmed somewhere else on the table, in another direction."""
    r = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    out = dict(c)
    for k in ("ee_pos", "obj_pos"):
        out[k] = c[k] @ r.T + np.array([*shift, 0])
    return out


def test_path_is_the_same_wherever_and_whichever_way_the_clip_was_filmed():
    c = place_clip()
    a = skill_ref.canonical(c, moves=True)
    b = skill_ref.canonical(moved(c, (0.05, -0.1), 2.0), moves=True)
    assert np.allclose(a, b, atol=1e-9)
    # the cube ends at the origin of o and starts 20 cm back along the displacement (x), 14 cm up
    assert np.allclose(a[-1, 3:], 0)
    assert np.allclose(a[0, 3:], [-0.2, 0.0, 0.14], atol=1e-9)


def test_moves_flag_and_lift_displacement():
    places = [place_clip(start=(0.2 + 0.01 * i, 0.0)) for i in range(4)]
    lifts = [lift_clip(height=0.12 + 0.01 * i) for i in range(4)]
    assert skill_ref.skill_moves(places) and not skill_ref.skill_moves(lifts)
    ref = skill_ref.build(places + lifts + reach_and_push(), [f"c{i}" for i in range(12)])
    assert np.allclose(ref["pick_lift_delta"], [0, 0, 0.135], atol=1e-6)     # mean lift, read off the clips
    assert ref["pick_lift_moves"] == 0 and ref["place_down_moves"] == 1


def reach_and_push():
    n = 10
    t = np.linspace(0, 1, n)
    cube = np.tile([0.0, 0.0, CUBE_HALF], (n, 1))
    reach = {"t": t, "ee_pos": cube + np.outer(1 - t, [0.1, 0, 0.2]), "ee_yaw": np.zeros(n), "grip": np.zeros(n),
             "obj_pos": cube, "skill": "reach"}
    obj = cube + np.outer(t, [0.12, 0, 0])
    push = {"t": t, "ee_pos": obj - [0.04, 0, 0], "ee_yaw": np.zeros(n), "grip": np.full(n, 0.6), "obj_pos": obj,
            "skill": "push"}
    return [reach, dict(reach), push, dict(push)]


def test_sigma_follows_how_much_the_clips_vary():
    # two places that start 10 cm apart but end at the same spot: wide at the start, tight at the end
    clips = [place_clip(start=(0.15, 0.0)), place_clip(start=(0.25, 0.0))]
    ref = skill_ref.build(clips + [lift_clip(), lift_clip()] + reach_and_push(), [f"c{i}" for i in range(8)])
    sig = ref["place_down_sigma"]
    assert sig[0] > 0.04 and np.isclose(sig[-1], skill_ref.SIGMA_FLOOR)


def test_failed_clips_tighten_the_done_radius():
    good = [moved(place_clip(), (0, 0), 0)] * 3
    sloppy = dict(place_clip())                       # a success that ends with the hand 3 cm off
    sloppy["ee_pos"] = sloppy["ee_pos"] + np.array([0, 0, 0.03]) * (np.linspace(0, 1, 30) > 0.9)[:, None]
    fail = dict(place_clip())                         # a failure whose hand ends 2.5 cm off
    fail["ee_pos"] = fail["ee_pos"] + np.array([0, 0, 0.025]) * (np.linspace(0, 1, 30) > 0.9)[:, None]
    rest = [lift_clip(), lift_clip()] + reach_and_push()
    names = lambda k: [f"place_{i}" for i in range(k)]  # noqa: E731
    r1 = skill_ref.build(good + [sloppy] + rest, names(4) + [f"o{i}" for i in range(6)])
    r2 = skill_ref.build(good + [sloppy, fail] + rest, names(4) + ["place_x_fail"] + [f"o{i}" for i in range(6)])
    assert r1["place_down_done_radius"] >= 0.0225 - 1e-6       # the sloppy success sets it
    assert r2["place_down_done_radius"] < r1["place_down_done_radius"]
