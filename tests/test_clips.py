"""Clip validation and cube inference (sim/clips.py). Uses the stand-in clips when present.

    python -m pytest tests -q
"""
from pathlib import Path

import numpy as np
import pytest

from sim import clips
from sim.envs.tasks import CUBE_HALF

STANDIN = Path("data/processed/clips_standin")


def make_clip(skill="reach", n=20):
    t = np.arange(n) * 0.1
    ee = np.stack([np.linspace(0.0, 0.1, n), np.zeros(n), np.linspace(0.25, 0.03, n)], -1)
    return {"t": t, "ee_pos": ee, "ee_yaw": np.zeros(n), "grip": np.zeros(n),
            "obj_pos": np.full((n, 3), np.nan), "skill": skill}


def test_valid_clip_has_no_errors():
    errs, warns = clips.validate(make_clip())
    assert errs == []
    assert any("untracked" in w for w in warns)


@pytest.mark.parametrize("breakage, expected", [
    (lambda c: c.pop("grip"), "missing keys"),
    (lambda c: c.update(t=np.arange(20) * 0.033), "sample period"),
    (lambda c: c.update(ee_pos=c["ee_pos"] * 1000), "not metres"),
    (lambda c: c.update(skill="wave"), "unknown skill"),
    (lambda c: c.update(grip=np.full(20, 2.0)), "grip outside"),
])
def test_broken_clips_are_rejected(breakage, expected):
    c = make_clip()
    breakage(c)
    errs, _ = clips.validate(c)
    assert any(expected in e for e in errs), errs


def test_preroll_and_layout_for_a_held_cube():
    c = make_clip("place-down")
    c["grip"][:] = 1.0
    c["grip"][-3:] = 0.0
    c["ee_pos"][:, 2] = np.linspace(0.16, CUBE_HALF + 0.001, 20)
    plan = clips.preroll_plan(c)
    assert plan.shape == (clips.PRE_TICKS, 4)
    assert plan[-1, 3] == 0.0                     # ends holding the cube
    lay = clips.layout_for(c, np.random.default_rng(0))
    assert np.allclose(lay[:2], c["ee_pos"][0, :2])   # cube placed under the start pose


@pytest.mark.skipif(not STANDIN.exists(), reason="no stand-in clips")
@pytest.mark.parametrize("skill, tol", [("reach", 0.02), ("pick-lift", 0.02), ("place-down", 0.03), ("push", 0.04)])
def test_inferred_cube_matches_tracked_cube_on_standins(skill, tol):
    """Hide the tracked cube of stand-in clips and check the inference recovers it."""
    errs = []
    for f in sorted(STANDIN.glob(f"{skill}_*.npz"))[:20]:
        c = clips.load(f)
        truth = c["obj_pos"].copy()
        c["obj_pos"] = np.full_like(truth, np.nan)
        guess = clips.infer_object(c)
        errs.append(np.linalg.norm(guess - truth, axis=1).mean())
    print(f"{skill}: mean inferred-cube error {np.mean(errs) * 100:.1f} cm")
    assert np.mean(errs) < tol
