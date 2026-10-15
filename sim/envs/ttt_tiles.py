"""Flat tic-tac-toe pieces drawn on the table (15 Oct, user: "a scene where tic-tac-toe is done ... real time"). Visual only: thin coloured
prims with no collision, so the physics, the teacher (privileged state) and every check are unchanged; only the cameras see them.

  grid  2 + 2 light lines between the 9 squares (tasks.BOARD_C, tasks.PITCH)
  X     two orange strokes; O a blue disc; at most 5 of each per scene

The prims are scene assets (spawned in every env before the simulation starts: prims added to a running stage broke the wrist camera's
image); unused ones wait 1 m under the table. Used by play/tictactoe.py (the live game) and, with VLA_TTT_TILES=1, by the keypoint env
(random pieces on the squares other than the target at every reset, for recording / evaluating a student that has seen them)."""
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from sim.envs import tasks

SQUARES = np.array([[tasks.BOARD_C[0] + dx * tasks.PITCH, tasks.BOARD_C[1] + dy * tasks.PITCH] for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
                   np.float32)                       # index = row * 3 + col as play/games/ttt.py
_L, _W, _H = 3 * tasks.PITCH, 0.004, 0.002
POOL = {"line_x": 2, "line_y": 2, "bar": 10, "o": 5}
_C8, _S8 = float(np.cos(np.pi / 8)), float(np.sin(np.pi / 8))
ROT = {"flat": (1.0, 0.0, 0.0, 0.0), "x1": (_C8, 0.0, 0.0, _S8), "x2": (_C8, 0.0, 0.0, -_S8)}


def _flat(size, rgb):
    return sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))


PROTO = {"line_x": _flat((_L, _W, _H), (0.85, 0.85, 0.82)),          # grid line along the table's x
         "line_y": _flat((_W, _L, _H), (0.85, 0.85, 0.82)),          # along y
         "bar": _flat((0.065, 0.013, 0.003), (0.95, 0.45, 0.10)),    # one stroke of an X
         "o": sim_utils.CylinderCfg(radius=0.027, height=0.003, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.90)))}


def add_assets(scene_cfg):
    """Register the pieces as scene assets (call before the env is built)."""
    for kind, cnt in POOL.items():
        for j in range(cnt):
            setattr(scene_cfg, f"ttt_{kind}{j}", AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/TTT_{kind}{j}", spawn=PROTO[kind],
                                                               init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0))))


class Tiles:
    """Moves the pieces of env i to show a board (list of 9: None / "X" / "O")."""

    def __init__(self, env):
        from pxr import Gf
        self.Gf = Gf
        stage = sim_utils.get_current_stage()
        self.n = env.num_envs
        self.table = env.table_origin.cpu().numpy()
        self.env0 = env.scene.env_origins.cpu().numpy()
        self.pool = {(i, k): [stage.GetPrimAtPath(f"/World/envs/env_{i}/TTT_{k}{j}") for j in range(c)]
                     for i in range(self.n) for k, c in POOL.items()}

    def _place(self, i, prim, xyz, rot):
        Gf = self.Gf
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*[float(v) for v in np.asarray(xyz) - self.env0[i]]))
        q = prim.GetAttribute("xformOp:orient")
        if "quatf" in str(q.GetTypeName()).lower():
            q.Set(Gf.Quatf(float(rot[0]), Gf.Vec3f(*[float(v) for v in rot[1:]])))
        else:
            q.Set(Gf.Quatd(float(rot[0]), Gf.Vec3d(*[float(v) for v in rot[1:]])))

    def show(self, i, board, grid=True):
        o, used = self.table[i], {k: 0 for k in POOL}

        def put(kind, xyz, rot):
            if used[kind] < len(self.pool[i, kind]):
                self._place(i, self.pool[i, kind][used[kind]], o + np.array(xyz), rot)
                used[kind] += 1
        if grid:
            c0, p0 = tasks.BOARD_C, tasks.PITCH
            for k in (-0.5, 0.5):                     # the two inner lines each way
                put("line_y", [c0[0] + k * p0, c0[1], _H / 2], ROT["flat"])
                put("line_x", [c0[0], c0[1] + k * p0, _H / 2], ROT["flat"])
        for q, v in enumerate(board):
            if v == "X":
                put("bar", [SQUARES[q][0], SQUARES[q][1], 0.0015], ROT["x1"])
                put("bar", [SQUARES[q][0], SQUARES[q][1], 0.0015], ROT["x2"])
            elif v == "O":
                put("o", [SQUARES[q][0], SQUARES[q][1], 0.0015], ROT["flat"])
        for kind in POOL:                             # the rest back under the table
            for prim in self.pool[i, kind][used[kind]:]:
                self._place(i, prim, [o[0], o[1], -1.0], ROT["flat"])

    def randomize(self, i, target_xy, rng):
        """A random position of a game in progress: 0-7 pieces (X / O alternating, no more than 5 of each) on squares other than the one
        under the target pad; the grid in 85% of the scenes."""
        free = [q for q in range(9) if np.linalg.norm(SQUARES[q] - np.asarray(target_xy)) > 0.02]
        k = int(rng.integers(0, min(7, len(free)) + 1))
        board = [None] * 9
        first = "XO"[int(rng.integers(2))]
        for j, q in enumerate(rng.permutation(free)[:k]):
            board[int(q)] = first if j % 2 == 0 else ("O" if first == "X" else "X")
        self.show(i, board, grid=rng.random() < 0.85)
        return board
