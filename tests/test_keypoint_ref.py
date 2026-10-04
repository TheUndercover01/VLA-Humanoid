"""Clips -> keypoint waypoints (motion/keypoint_ref.py) on hand-built clips.

    python -m pytest tests -q
"""
import numpy as np

from motion import keypoint_ref as kr
from sim.envs.tasks import CUBE_HALF


def yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)


def clip(skill, pos, yaw=None, n=None):
    pos = np.asarray(pos, float)
    yaw = np.zeros(len(pos)) if yaw is None else np.asarray(yaw, float)
    return {"t": np.arange(len(pos)) * 0.1, "ee_pos": pos.copy(), "ee_yaw": np.zeros(len(pos)),
            "grip": np.ones(len(pos)), "obj_pos": pos, "obj_quat": yaw_quat(yaw), "skill": skill}


def carry(start=(0.2, 0.0), end=(0.0, 0.0), height=0.14, n=30):
    t = np.linspace(0, 1, n)
    xy = np.array(start) + (np.array(end) - np.array(start)) * np.clip(t / 0.6, 0, 1)[:, None]
    z = CUBE_HALF + height * np.clip((0.9 - t) / 0.3, 0, 1)
    return np.column_stack([xy, z])


def moved(c, shift, yaw):
    """The same clip filmed elsewhere on the table and in another direction (cube turned with it)."""
    r = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    out = dict(c)
    out["obj_pos"] = c["obj_pos"] @ r.T + np.array([*shift, 0])
    w, z = c["obj_quat"][:, 0], c["obj_quat"][:, 3]
    out["obj_quat"] = yaw_quat(2 * np.arctan2(z, w) + yaw)
    return out


def test_corners_of_an_upright_cube():
    k = kr.corners(np.array([[0.1, 0.2, CUBE_HALF]]), yaw_quat(np.array([0.0])))[0]
    assert np.allclose(k.min(0), [0.1 - CUBE_HALF, 0.2 - CUBE_HALF, 0.0])
    assert np.allclose(k.max(0), [0.1 + CUBE_HALF, 0.2 + CUBE_HALF, 2 * CUBE_HALF])
    k45 = kr.corners(np.array([[0, 0, CUBE_HALF]]), yaw_quat(np.array([np.pi / 4])))[0]
    assert np.isclose(np.abs(k45[:, 0]).max(), CUBE_HALF * np.sqrt(2))       # a turned cube's corners stick out


def test_waypoints_do_not_depend_on_where_or_which_way_the_clip_was_filmed():
    c = clip("place-down", carry())
    a, da = kr.clip_waypoints(c, moves=True)
    b, db = kr.clip_waypoints(moved(c, (0.05, -0.1), 2.0), moves=True)
    assert np.allclose(a, b, atol=1e-9) and np.allclose(da, db, atol=1e-9)
    assert np.allclose(a[-1], 0)                                              # the last waypoint is the goal pose
    assert np.allclose(a[0].mean(0), [-0.2, 0.0, 0.14], atol=1e-9)           # starts 20 cm back along the travel, 14 cm up


def test_a_cube_turned_during_the_clip_shows_up_in_the_corners():
    pos = carry()
    still = kr.clip_waypoints(clip("place-down", pos), True)[0]
    turned = kr.clip_waypoints(clip("place-down", pos, yaw=np.linspace(0.5, 0, len(pos))), True)[0]
    # same centre path, but the corners of the turned clip spread around it at the start
    assert np.allclose(still[0].mean(0), turned[0].mean(0), atol=1e-9)
    assert np.abs(turned[0] - turned[0].mean(0)).max() > np.abs(still[0] - still[0].mean(0)).max() + 0.005


def lift_clip(h):
    z = CUBE_HALF + h * np.clip(np.linspace(-1, 1, 20), 0, 1)
    return clip("pick-lift", np.column_stack([np.full(20, 0.0), np.full(20, 0.1), z]))


def push_clip(d):
    x = np.linspace(0, d, 20)
    return clip("push", np.column_stack([x, np.zeros(20), np.full(20, CUBE_HALF)]))


def test_lift_height_and_moves_flag_come_from_the_clips():
    clips = [lift_clip(0.12 + 0.01 * i) for i in range(4)] + [push_clip(0.14)] * 2 + \
            [clip("place-down", carry(start=(0.2 + 0.01 * i, 0))) for i in range(3)]
    ref = kr.build(clips)
    assert np.allclose(ref["pick_lift_delta"], [0, 0, 0.135], atol=1e-6)
    assert not ref["pick_lift_moves"] and ref["push_moves"] and ref["place_down_moves"]


def test_spread_is_wide_where_the_clips_differ_and_tight_at_the_goal():
    clips = [lift_clip(0.14)] * 2 + [push_clip(0.14)] * 2 + \
            [clip("place-down", carry(start=(0.12, 0))), clip("place-down", carry(start=(0.28, 0)))]
    sp = kr.build(clips)["place_down_spread"]
    assert sp[0] > 0.05 and np.isclose(sp[-1], kr.SPREAD_FLOOR)
