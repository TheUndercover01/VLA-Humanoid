"""Video of how keypoint waypoints are made from the clips and laid over every new command in sim.

Part 1 (clips -> template): the place-down clips' cube paths as filmed; the same paths relative to
where each cube ends and turned to its direction of travel; 20 resampled waypoints averaged, with the
clips' spread. Part 2 (template -> envs): commands recorded by sim/record_waypoints.py: the layout,
the waypoints the env placed for that command, and the cube's track (reached waypoints light up).
Last: every placed place-down path, back in goal-relative coordinates.

    python -m analysis.waypoint_video --clips data/processed/clips_standin --rec runs/viz/waypoints.npz \
        --out media/18_waypoints_from_clips_to_sim.mp4
"""
import argparse
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from motion import keypoint_ref as kr  # noqa: E402
from sim.clips import load  # noqa: E402

FPS = 10
SKILL_NAME = {1: "push", 2: "pick-lift", 3: "place-down"}
CUBE_NAME = {0: "red", 1: "blue"}
DEST_NAME = {0: "the green target", 1: "the other cube", 2: ""}
COL = {0: "tab:red", 1: "tab:blue"}


def top_square(c, ax, **kw):
    """Top-view outline of a cube from its 8 corners (the 4 highest, ordered round the centre)."""
    c = np.asarray(c)
    top = c[np.argsort(c[:, 2])[-4:], :2]
    ang = np.arctan2(*(top - top.mean(0)).T[::-1])
    p = top[np.argsort(ang)] * 100
    ax.fill(p[:, 0], p[:, 1], **kw)


class Canvas:
    def __init__(self):
        self.fig, (self.top, self.side) = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100,
                                                       gridspec_kw={"width_ratios": [1, 1.25]})
        self.frames = []

    def start(self, title, subtitle=""):
        for ax in (self.top, self.side):
            ax.cla()
            ax.grid(alpha=0.3)
        self.fig.text(0.5, 0.96, title, ha="center", fontsize=14)
        if subtitle:
            self.fig.text(0.5, 0.92, subtitle, ha="center", fontsize=10, color="0.3")

    def grab(self, n=1):
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        self.frames.extend([img] * n)
        for t in list(self.fig.texts):
            t.remove()


def canonical(c):
    """Clip cube centre in the goal-relative, travel-aligned frame (what the waypoints are built in)."""
    pos = np.asarray(c["obj_pos"], float)
    d = pos[-1, :2] - pos[0, :2]
    return kr.rot_z(pos - pos[-1], np.arctan2(d[1], d[0]))


def part1(cv, clips):
    raw = [np.asarray(c["obj_pos"], float) * 100 for c in clips]
    can = [canonical(c) * 100 for c in clips]
    T = max(len(r) for r in raw)
    sub = "stand-in place-down clips (scripted robot in sim); with phone clips the cube pose comes from an ArUco marker"
    # 1a: the clips as filmed
    for t in range(1, T + 1, 1):
        cv.start("1. The clips: where the cube went in each place-down clip (table frame)", sub)
        for r in raw:
            k = min(t, len(r))
            cv.top.plot(r[:k, 0], r[:k, 1], color="tab:red", alpha=0.6)
            cv.side.plot(np.arange(k) / 10, r[:k, 2], color="tab:red", alpha=0.6)
        cv.top.set(xlim=(-20, 25), ylim=(-25, 25), xlabel="table x (cm)", ylabel="table y (cm)", title="seen from above",
                   aspect="equal")
        cv.side.set(xlim=(0, 3.6), ylim=(0, 22), xlabel="time (s)", ylabel="cube height (cm)",
                    title="height over time: lifted, carried, lowered")
        cv.grab()
    cv.grab(10)
    # 1b: relative to the end pose, turned to the travel direction
    for a in np.linspace(0, 1, 25):
        cv.start("2. Same paths, relative to where each cube ends and turned to its direction of travel",
                 "every path now ends at (0, 0, 0); before the end they say 'this far back, this high'")
        for r, cn in zip(raw, can):
            r0 = r - (r[-1] * [1, 1, 0])          # slide the end over the origin first, then turn
            p = (1 - a) * r0 + a * cn
            cv.top.plot(p[:, 0], p[:, 1], color="tab:red", alpha=0.6)
            cv.side.plot(p[:, 0], p[:, 2], color="tab:red", alpha=0.6)
        cv.top.plot(0, 0, "k*", ms=12)
        cv.side.plot(0, 2.5, "k*", ms=12)
        cv.top.set(xlim=(-35, 20), ylim=(-25, 25), xlabel="along travel (cm)", ylabel="sideways (cm)",
                   title="from above", aspect="equal")
        cv.side.set(xlim=(-35, 5), ylim=(-3, 22), xlabel="along travel, 0 = where it ends (cm)",
                    ylabel="height above where it ends (cm)", title="from the side")
        cv.grab()
    cv.grab(10)
    # 1c: 20 waypoints per clip, averaged, with spread
    wps = np.stack([kr.resample(cn) for cn in can])
    mean = wps.mean(0)
    spread = np.sqrt(((wps - mean) ** 2).sum(-1).mean(0))
    for step in range(3):
        cv.start("3. Resample each clip to 20 points and average: the place-down template",
                 ["20 evenly spaced points on every clip", "their average: 20 waypoints",
                  "circles: how much the clips differ there = how close the robot must get"][step])
        for cn, w in zip(can, wps):
            cv.side.plot(cn[:, 0], cn[:, 2], color="tab:red", alpha=0.15)
            cv.side.plot(w[:, 0], w[:, 2], ".", color="tab:red", alpha=0.5, ms=4)
            cv.top.plot(w[:, 0], w[:, 1], ".", color="tab:red", alpha=0.3, ms=3)
        if step >= 1:
            cv.side.plot(mean[:, 0], mean[:, 2], "ko-", ms=5, lw=2)
            cv.top.plot(mean[:, 0], mean[:, 1], "ko-", ms=5, lw=2)
            for i, m in enumerate(mean):
                cv.side.annotate(str(i), (m[0], m[2] + 0.8), fontsize=7, ha="center")
        if step >= 2:
            for m, s in zip(mean, spread):
                cv.side.add_patch(plt.Circle((m[0], m[2]), max(2 * s, 1.0), fill=False, color="tab:green", alpha=0.6))
        cv.top.set(xlim=(-35, 20), ylim=(-25, 25), xlabel="along travel (cm)", ylabel="sideways (cm)",
                   title="from above", aspect="equal")
        cv.side.set(xlim=(-35, 5), ylim=(-3, 22), xlabel="along travel, 0 = goal (cm)", ylabel="height above goal (cm)",
                    title="from the side", aspect="equal")
        cv.grab(25)


def part2(cv, rec, n_per_skill=4):
    st = {k[6:]: rec[k] for k in rec.files if k.startswith("start_")}
    track, reached = rec["track"], rec["reached"]
    nsteps = track.shape[1]
    chosen = []
    for skill in (3, 2, 1):
        cands = []
        for j in np.nonzero(st["skill"] == skill)[0]:
            e, s0 = st["env"][j], st["step"][j]
            later = [st["step"][k] for k in np.nonzero(st["env"] == e)[0] if st["step"][k] > s0]
            s1 = min(later) if later else nsteps
            if s1 - s0 >= 15 and s0 > 0 or (s1 - s0 >= 15 and skill != 3):
                cands.append((j, s0, s1))
        chosen += cands[:n_per_skill]
    for n, (j, s0, s1) in enumerate(chosen):
        e, skill, cube, dest = (int(st[k][j]) for k in ("env", "skill", "cube", "dest"))
        path, goal, start, tgt, other = st["path"][j], st["goal"][j], st["start"][j], st["target"][j], st["other"][j]
        centres = path.mean(1) * 100
        g = goal.mean(0) * 100
        travel = g[:2] - centres[0, :2]
        u = travel / max(np.linalg.norm(travel), 1e-6)

        def along(p):                       # horizontal distance to the goal along this command's travel
            return ((p[..., :2] - g[:2]) * u).sum(-1) if np.linalg.norm(travel) > 1 else np.zeros(p.shape[:-1])

        what = f"{SKILL_NAME[skill]} {CUBE_NAME[cube]}" + (f" onto {DEST_NAME[dest]}" if skill != 2 else " (goal: start + clips' lift)")
        tr = track[e, s0:s1] * 100
        for t in range(0, s1 - s0, 1):
            cv.start(f"Env {e}, command start at t = {s0 / 10:.1f} s: {what}  ({n + 1}/{len(chosen)})",
                     "the env lays the same template over this command's goal, stretched so waypoint 0 is the cube's start")
            ax = cv.top
            ax.add_patch(plt.Rectangle(((tgt[0] - 0.03) * 100, (tgt[1] - 0.03) * 100), 6, 6, color="tab:green", alpha=0.35))
            oc = kr.corners(other[None], np.array([[1.0, 0, 0, 0]]))[0]
            top_square(oc, ax, color=COL[1 - cube], alpha=0.35)
            top_square(start, ax, color=COL[cube], alpha=0.25)
            top_square(goal, ax, fill=False, edgecolor="k", lw=1.5, ls="--")
            r = reached[e, s0 + t]
            for i, w in enumerate(path):
                top_square(w, ax, fill=False, edgecolor="tab:orange" if i <= r else "0.6", lw=0.6)
            ax.plot(centres[:, 0], centres[:, 1], "-", color="0.5", lw=1)
            ax.plot(tr[:t + 1, 0], tr[:t + 1, 1], color=COL[cube], lw=2)
            ax.plot(*tr[t, :2], "o", color=COL[cube], ms=8)
            ax.set(xlim=(-25, 30), ylim=(-30, 30), xlabel="table x (cm)", ylabel="table y (cm)", aspect="equal",
                   title="above: target, cubes, placed waypoints (orange = reached)")
            sd = cv.side
            sd.plot(along(centres), centres[:, 2], "o-", color="0.6", ms=4)
            sd.plot(along(centres[:r + 1]), centres[:r + 1, 2], "o", color="tab:orange", ms=6)
            sd.plot(along(tr[:t + 1]), tr[:t + 1, 2], color=COL[cube], lw=2)
            sd.plot(along(tr[t]), tr[t, 2], "o", color=COL[cube], ms=8)
            sd.plot(0, g[2], "k*", ms=12)
            sd.set(xlim=(-35, 8), ylim=(0, 25), xlabel="distance to the goal along the travel (cm)",
                   ylabel="cube height (cm)", title=f"side: waypoint centres vs the cube (reached {r}/19)")
            cv.grab()
        cv.grab(8)
    # every placed place-down path, in goal-relative travel coordinates
    cv.start("Every place-down command in the recording: placed waypoints back in goal-relative coordinates",
             "same template every time, stretched to that command's start distance (black: the clips' template)")
    for j in np.nonzero(st["skill"] == 3)[0]:
        c = st["path"][j].mean(1) * 100
        g = st["goal"][j].mean(0) * 100
        d = g[:2] - c[0, :2]
        cr = kr.rot_z(c - g, np.arctan2(d[1], d[0]))
        cv.side.plot(cr[:, 0], cr[:, 2], "-", color="tab:orange", alpha=0.5)
        cv.top.plot(cr[:, 0], cr[:, 1], "-", color="tab:orange", alpha=0.5)
    tm = rec["template_place"].mean(1) * 100
    cv.side.plot(tm[:, 0], tm[:, 2], "ko-", lw=2, ms=4)
    cv.top.plot(tm[:, 0], tm[:, 1], "ko-", lw=2, ms=4)
    cv.top.set(xlim=(-45, 10), ylim=(-25, 25), xlabel="along travel (cm)", ylabel="sideways (cm)", aspect="equal",
               title="from above")
    cv.side.set(xlim=(-45, 8), ylim=(-3, 25), xlabel="along travel, 0 = goal (cm)", ylabel="height above goal (cm)",
                title=f"from the side ({int((st['skill'] == 3).sum())} commands)")
    cv.grab(40)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", default="data/processed/clips_standin")
    p.add_argument("--rec", default="runs/viz/waypoints.npz")
    p.add_argument("--out", default="media/18_waypoints_from_clips_to_sim.mp4")
    p.add_argument("--n_clips", type=int, default=12)
    a = p.parse_args()
    clips = [load(f) for f in sorted(Path(a.clips).glob("place-down_*.npz"))][:a.n_clips]
    cv = Canvas()
    part1(cv, clips)
    part2(cv, np.load(a.rec))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, cv.frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    print(f"wrote {a.out}: {len(cv.frames)} frames, {len(cv.frames) / FPS:.0f} s")


if __name__ == "__main__":
    main()
