"""Robot Playground v2 (10 Oct, user: anyone plays with the robot on my website; games: tic-tac-toe, tower duel).

One Isaac process with --slots scenes (one per player) and a web server in a thread. There is ONE policy for everything (the student of
sim/envs/multi_env.py's game mixture, or --backend teacher for tests): it only ever hears "pick up X, then put X on the target / on Y", with
the "(done)" marks it writes itself. The games are different uses of it:
  build   the player chains skill buttons into a sentence (up to 10 commands) and watches the robot run it;
  ttt     tic-tac-toe: ttt.py decides the robot's square, the green pad is moved there, the sentence is "pick up the cube, then put the cube
          on the target"; the human's O (a cylinder) is placed on the clicked square by the game; a fumbled move is a lost turn;
  tower   tower duel: the human adds a tile with an offset of their choice, the robot adds a disc on top ("pick up the disc, then put the
          disc on the tile"); whoever's piece makes the tower fall loses.
The simulator's own check of what really happened is shown next to what the robot believes.

Run (the policy server in the lerobot env, this in the Isaac env; both env vars of training must be set, see play/run_playground.sh):
  python -m vla.server --policy <ckpt> --port 6201
  python play/playground.py --headless --device cuda:0 --vla_port 6201 --port 8800 [--backend teacher --teacher runs/rl/<run>/model_N.pt]
Binds 127.0.0.1 only.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vla_port", type=int, default=6201)
parser.add_argument("--port", type=int, default=8800, help="web server port (127.0.0.1)")
parser.add_argument("--slots", type=int, default=4, help="scenes = players at the same time")
parser.add_argument("--session_s", type=float, default=600.0, help="idle seconds a player keeps a slot (the clock restarts at every action)")
parser.add_argument("--site", default="../ayushdeshmukh", help="the website folder, served under /site/ so ONE port carries the page and the robot")
parser.add_argument("--fps", type=float, default=10.0, help="frames sent per second per player")
parser.add_argument("--backend", default="vla", choices=["vla", "teacher"], help="teacher = the privileged RL policy (tests only; labelled as such on the page)")
parser.add_argument("--teacher", default=None, help="rsl_rl checkpoint for --backend teacher")
parser.add_argument("--ref", default="data/processed/keypoint_ref_real_straight.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import asyncio  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from multiprocessing.connection import Client  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.clips import SKILLS  # noqa: E402
from sim.envs import keypoint_env as ke  # noqa: E402
from sim.envs import tasks  # noqa: E402
from sim.envs.multi_env import (HZ, NONE, PICK, PLACE, PUSH, SQUARES, SUPPLY, TGT, PandaMultiEnv, PandaMultiEnvCfg, board_spec, make_spec,  # noqa: E402
                                park_pos, push_ok, sample_positions, static_park)
from play.games import ttt, tower  # noqa: E402
from vla.prompts import command_prompt, marked_sentence  # noqa: E402

AUTHKEY = b"vla-humanoid"
MAX_CMDS = 10
DONE_THRESH, DONE_HOLD = 0.85, 5           # the eval defaults (sim/eval_chains.py)
SETTLE = 15                                # ticks (1.5 s) after the robot says it is finished before the result is read
RUN_TIMEOUT = 450                          # ticks (45 s) of one run
IDLE_PAD = np.array([0.03, -0.19], np.float32)
TOWER_XY = np.array([0.03, -0.04], np.float32)
STATIC_FIRST = {"cube": 0, "cyl": 4, "disc": 8, "tile": 11}
STATIC = Path(__file__).parent / "static"
JS2ENV_DEST = {0: TGT, 2: NONE}            # the page's destination codes: 0 target, 1 the other object, 2 none


def to_env(chain):
    """The page's [skill, obj, dest] (dest 0 target / 1 the other of the two / 2 none) -> the multi env's codes."""
    return [(SKILLS.index(k), o, (1 - o) if d == 1 else JS2ENV_DEST[d]) for k, o, d in chain]


def check_chain(chain):
    """Rules of the sentence builder (as eval_chains.random_chain): a push only first, the held object put on the free target or on the
    other one, pick up only with an empty hand and nothing on top. Returns (list of (skill, obj, dest) in the page's codes, None) or (None, why)."""
    if not isinstance(chain, list) or not 1 <= len(chain) <= MAX_CMDS:
        return None, f"1 to {MAX_CMDS} commands"
    pos, held, out = {0: "table", 1: "table"}, None, []
    for i, c in enumerate(chain):
        if not (isinstance(c, list) and len(c) == 3 and c[0] in ("push", "pick-lift", "place-down") and c[1] in (0, 1) and c[2] in (0, 1, 2)):
            return None, f"command {i + 1} is malformed"
        k, o, d = c
        free_target = "target" not in pos.values()
        if k == "push":
            if i or d != 0:
                return None, "a push can only be the first command"
            pos[o] = "target"
        elif k == "pick-lift":
            if held is not None or d != 2 or pos[1 - o] == "on":
                return None, f"command {i + 1}: cannot pick that up now"
            held = o
        else:
            if held != o or d == 2 or (d == 0 and not free_target and pos[o] != "target"):
                return None, f"command {i + 1}: cannot put that there now"
            pos[o], held = ("target" if d == 0 else "on"), None
        out.append((k, o, d))
    return out, None


class Slot:
    def __init__(self):
        self.owner = None            # session id
        self.claimed = 0.0
        self.mode = "build"
        self.g = {}                  # game state of the mode
        self.chain = None            # env-coded chain being run, or None
        self.shown = None            # the last finished chain (build mode keeps it on screen)
        self.stage = self.streak = 0
        self.done_p = 0.0
        self.t_run = 0
        self.settle = -1             # >= 0: finished, counting ticks until the result is read
        self.lhold, self.reached, self.snap = 0, False, None
        self.note = ""
        self.frame = None            # latest jpeg (front | wrist side by side)


def main():
    cfg = PandaMultiEnvCfg(task="c1", cameras=True, image_size=256, terminate_on_success=False, episode_s=1e6, hold_fix=True, derived=False,
                           action_filter=0.95 if args.backend == "teacher" else 0.0, hand_leg=True, ref_path=args.ref, max_chain=2)
    cfg.scene.num_envs = args.slots
    cfg.sim.device = args.device
    env = PandaMultiEnv(cfg)
    dev, n = env.device, args.slots
    rng = np.random.default_rng(int(time.time()))
    py_rng = random.Random(int(time.time()))
    env.set_specs([board_spec(rng, n_stat=0, c=0, other_on=False) for _ in range(n)])
    env.reset()
    # the games decide when an episode ends: no automatic reset of a scene (a knocked piece, a long episode)
    _dones = env._get_dones

    def no_reset():
        _dones()
        z = torch.zeros(n, dtype=torch.bool, device=dev)
        return z, z.clone()
    env._get_dones = no_reset
    if args.backend == "vla":
        conn = Client(("localhost", args.vla_port), authkey=AUTHKEY)
        teacher = None
    else:
        from sim.rl_cfg import RslPolicy
        conn, teacher = None, RslPolicy(args.teacher, env)
    slots = [Slot() for _ in range(n)]
    lock = threading.Lock()
    inbox = queue.Queue()
    hold = torch.zeros(n, 5, device=dev)
    hold[:, 4] = 1.0                                           # idle: stay still, fingers open
    zeros6 = torch.zeros(1, 6, device=dev)

    # ---------------- scene helpers (sim thread) ----------------
    def put(obj, i, xyz):
        pose = torch.tensor([[xyz[0], xyz[1], xyz[2], 1, 0, 0, 0]], dtype=torch.float32, device=dev)
        pose[:, :3] += env.table_origin[i:i + 1]
        e = torch.tensor([i], device=dev)
        obj.write_root_pose_to_sim(pose, env_ids=e)
        obj.write_root_velocity_to_sim(zeros6, env_ids=e)

    def pad_to(i, xy):
        put(env.target, i, (float(xy[0]), float(xy[1]), 0.001))
        env.tgt_now[i] = torch.tensor(np.asarray(xy, np.float32), device=dev)

    def piece_to(i, t, xy, z=None):
        put(env.objs[t], i, (float(xy[0]), float(xy[1]), HZ[t] + 0.001 if z is None else z))
        env.active[i, t] = True

    def park_piece(i, t):
        p = park_pos(t)
        put(env.objs[t], i, (float(p[0]), float(p[1]), float(p[2])))
        env.active[i, t] = False

    def static_to(i, q, xy, z):
        put(env.statics[q], i, (float(xy[0]), float(xy[1]), float(z)))
        env.static_active[i, q] = True
        env.static_start[i, q] = torch.tensor([float(xy[0]), float(xy[1]), float(z)], device=dev)

    def static_off(i, q):
        p = static_park(q)
        put(env.statics[q], i, (float(p[0]), float(p[1]), float(p[2])))
        env.static_active[i, q] = False

    def reset_scene(i, spec):
        """A fresh scene for slot i (the arm goes home): spec = a multi_env spec."""
        env.eval_specs[i] = spec
        ids = torch.tensor([i], device=dev)
        env._reset_idx(ids)
        env.episode_length_buf[ids] = 0
        hold[i] = 0.0
        hold[i, 4] = 1.0

    def begin_run(i, chain):
        """Give slot i the chain (env codes) as its command list; the robot starts from where its arm is."""
        s = slots[i]
        ids = torch.tensor([i], device=dev)
        cm = torch.zeros(1, ke.MAX_CMDS, 3, dtype=torch.long)
        cm[0, :len(chain)] = torch.tensor(chain)
        env.cmds[ids] = cm.to(dev)
        env.n_cmd[ids] = len(chain)
        for name in ("cur", "completed", "chold"):
            getattr(env, name)[ids] = 0
        for name in ("last_done", "succ_latch", "done_ok", "knocked", "static_hit"):
            getattr(env, name)[ids] = False
        env.need_start[ids] = True
        env.lift_start[ids] = torch.nan
        env.episode_length_buf[ids] = 0
        s.chain, s.shown, s.stage, s.streak, s.done_p, s.t_run, s.settle = chain, None, 0, 0, 0.0, 0, -1
        s.lhold, s.reached, s.snap = 0, False, env.obj_pos()[i].clone()

    def outcome_now(i, cmd, start):
        k, o, d = cmd
        P = env.obj_pos()[i]
        me = P[o]
        if k == PICK:
            return bool(me[2] - start[o, 2] > 0.05)
        if d < 6:
            return bool(((me[:2] - P[d, :2]).norm() < 0.03) & ((me[2] - P[d, 2] - HZ[d] - HZ[o]).abs() < 0.015))
        return bool(((me[:2] - env.tgt_now[i]).norm() < 0.03) & (me[2] < HZ[o] + 0.02))

    # ---------------- build mode ----------------
    def build_new(i):
        pos = None
        while pos is None:
            pos = sample_positions(rng, [0, 1])
        reset_scene(i, make_spec([(PICK, 0, NONE)], [0, 1], pos))
        pad_to(i, np.array(tasks.FIXED_TARGET, np.float32))
        slots[i].mode, slots[i].g, slots[i].chain, slots[i].shown = "build", {}, None, None
        for q in range(env.n_static):
            static_off(i, q)

    def build_run(i, chain_js):
        chain = to_env(chain_js)
        pos = None
        for _ in range(300):
            pos = sample_positions(rng, [0, 1])
            if pos is not None and (chain[0][0] != PUSH or push_ok(pos, chain[0][1], [0, 1], np.array(tasks.FIXED_TARGET))):
                break
        reset_scene(i, make_spec(chain, [0, 1], pos))
        pad_to(i, np.array(tasks.FIXED_TARGET, np.float32))
        begin_run(i, chain)

    # ---------------- tic-tac-toe ----------------
    def ttt_new(i, diff):
        reset_scene(i, board_spec(rng, n_stat=0, c=0, other_on=False))
        pad_to(i, IDLE_PAD)
        park_piece(i, 1)
        slots[i].mode = "ttt"
        slots[i].g = {"board": [None] * 9, "nX": 0, "nO": 0, "diff": diff if diff in ttt.DIFFICULTY else ttt.DEFAULT_DIFFICULTY,
                      "turn": "human", "result": "", "note": "Your move: click a square (you are O, the robot is X).", "sq": None, "tries": 0}
        slots[i].chain, slots[i].shown = None, None

    def ttt_human(i, k):
        g = slots[i].g
        if g["result"] or g["turn"] != "human" or not isinstance(k, int) or not 0 <= k < 9 or g["board"][k]:
            return
        static_to(i, STATIC_FIRST["cyl"] + g["nO"], SQUARES[k], HZ[1] + 0.001)
        g["nO"] += 1
        g["board"][k] = "O"
        end = ttt.winner(g["board"])
        if end:
            ttt_over(i, end)
        else:
            g["turn"], g["note"] = "robot", "The robot is thinking..."
            ttt_robot(i)

    def ttt_over(i, end):
        g = slots[i].g
        g["result"] = {"X": "robot", "O": "human", "draw": "draw"}[end]
        g["note"] = {"robot": "The robot wins.", "human": "You win!", "draw": "A draw."}[g["result"]]

    def ttt_robot(i):
        g = slots[i].g
        g["sq"] = ttt.choose(g["board"], "X", ttt.DIFFICULTY[g["diff"]], py_rng)
        g["tries"] = 0
        ttt_try(i)

    def ttt_try(i):
        g = slots[i].g
        pad_to(i, SQUARES[g["sq"]])
        piece_to(i, 0, SUPPLY)                                   # a fresh cube waits at the supply pad
        slots[i].note = ""
        begin_run(i, [(PICK, 0, NONE), (PLACE, 0, TGT)])

    def ttt_run_end(i, timed_out):
        s = slots[i]
        g = s.g
        s.chain = None
        cube = env.obj_pos()[i, 0]
        sq_xy = torch.tensor(SQUARES[g["sq"]], device=dev)
        placed = (not timed_out) and bool((cube[:2] - sq_xy).norm() < 0.04) and bool(cube[2] < 0.045)
        pad_to(i, IDLE_PAD)
        if placed:
            mine = STATIC_FIRST["cube"] + g["nX"]
            static_to(i, mine, cube[:2].cpu().numpy(), float(cube[2]))   # the cube stays; an identical fresh one waits at the supply
            g["nX"] += 1
            g["board"][g["sq"]] = "X"
            piece_to(i, 0, SUPPLY)
            g["note"] = ""
            for q in range(env.n_static):                        # a neighbour that was bumped is put back
                if env.static_active[i, q] and q != mine:
                    if (env.static_pos()[i, q, :2] - env.static_start[i, q, :2]).norm() > 0.03:
                        static_to(i, q, env.static_start[i, q, :2].cpu().numpy(), float(env.static_start[i, q, 2]))
                        g["note"] = "A neighbouring piece was bumped and put back. "
            end = ttt.winner(g["board"])
            if end:
                ttt_over(i, end)
            else:
                g["turn"], g["note"] = "human", g["note"] + "Your move."
        elif g["tries"] < 1:
            g["tries"] += 1
            ttt_try(i)
        else:
            piece_to(i, 0, SUPPLY)
            g["turn"], g["note"] = "human", "The robot fumbled the piece and loses its turn. Your move again."

    # ---------------- tower duel ----------------
    def tower_new(i):
        pos = np.zeros((tasks.N_TYPES, 2), np.float32)
        pos[3] = SUPPLY
        spec = make_spec([(PICK, 3, NONE), (PLACE, 3, 4)], [3], pos)
        spec["tgt"], spec["statics"], spec["kind"] = IDLE_PAD, [], 2
        reset_scene(i, spec)
        piece_to(i, 3, SUPPLY)
        for q in range(env.n_static):
            static_off(i, q)
        park_piece(i, 4)
        slots[i].mode = "tower"
        slots[i].chain, slots[i].shown = None, None
        slots[i].g = {"tw": tower.Tower(human_first=True), "refs": [], "rest": None, "phase": "human", "timer": 0, "result": "",
                      "note": "Your move: choose how far off-centre to put your tile, then press Place."}

    def tower_positions(i, refs):
        P, S = env.obj_pos()[i], env.static_pos()[i]
        return {j: tuple(float(x) for x in (P[v] if kind == "slot" else S[v]).cpu().numpy()) for j, (kind, v) in enumerate(refs)}

    def convert(i, g, t):
        """The piece of slot type t (disc 3 / tile 4) that sits on the tower becomes a static piece (an identical one), freeing the slot."""
        refs = g["refs"]
        for j, (kind, v) in enumerate(refs):
            if kind == "slot" and v == t:
                p = env.obj_pos()[i, t].cpu().numpy()
                q = next(q for q in range(env.n_static) if tasks.STATIC_TYPES[q] == t and not env.static_active[i, q])
                static_to(i, q, p[:2], float(p[2]))
                refs[j] = ("static", q)
        if t == 3:
            piece_to(i, 3, SUPPLY)
        else:
            park_piece(i, 4)

    def tower_human(i, dx, dy):
        g = slots[i].g
        if g["result"] or g["phase"] != "human":
            return
        tw = g["tw"]
        convert(i, g, 4)
        convert(i, g, 3)
        dx, dy = tw.clamp_offset(dx / 100.0, dy / 100.0)
        if g["refs"]:
            kind, v = g["refs"][-1]
            top = (env.static_pos()[i, v] if kind == "static" else env.obj_pos()[i, v]).cpu().numpy()
            base_xy, top_z = top[:2], float(top[2]) + 0.0125
        else:
            base_xy, top_z = TOWER_XY, 0.0
        piece_to(i, 4, base_xy + np.array([dx, dy], np.float32), top_z + 0.0125 + 0.006)
        g["refs"].append(("slot", 4))
        tw.place("human")
        g["phase"], g["timer"], g["rest"] = "settle_h", 0, None
        g["note"] = "Your tile is settling..."

    def tower_robot(i):
        g = slots[i].g
        g["phase"], g["note"] = "robot", "The robot adds a disc."
        begin_run(i, [(PICK, 3, NONE), (PLACE, 3, 4)])

    def tower_run_end(i, timed_out):
        s = slots[i]
        g = s.g
        s.chain = None
        kind, v = g["refs"][-1]
        top = (env.static_pos()[i, v] if kind == "static" else env.obj_pos()[i, v]).cpu().numpy()
        disc = env.obj_pos()[i, 3].cpu().numpy()
        on_top = (not timed_out) and np.linalg.norm(disc[:2] - top[:2]) < 0.035 and disc[2] > top[2] + 0.01
        if not on_top:                                           # the disc did not end up on the tower: a lost turn, the disc goes back to the supply
            piece_to(i, 3, SUPPLY)
            g["tw"].skip()
            g["phase"], g["note"] = "human", "The robot missed the tower and loses its turn. Your move again."
            return
        g["refs"].append(("slot", 3))
        if not g["tw"].over:
            g["tw"].place("robot")
        g["phase"], g["timer"] = "settle_r", 0
        g["note"] = "The disc is settling..."

    def tower_tick(i):
        """Every sim tick in a settle phase: after 1 s record where everything is, after 2.5 s compare."""
        g = slots[i].g
        g["timer"] += 1
        if g["timer"] == 10:
            g["rest"] = tower_positions(i, g["refs"])
        elif g["timer"] >= 25:
            who = "human" if g["phase"] == "settle_h" else "robot"
            fell = tower.has_fallen(tower_positions(i, g["refs"]), g["rest"])
            tw = g["tw"]
            if fell:
                tw.fell(who)
            if tw.over:
                g["result"] = {"robot_wins": "robot", "human_wins": "human", "draw": "draw"}[tw.over]
                g["note"] = {"robot": "Your tile made the tower fall: the robot wins.", "human": "The robot's disc made the tower fall: you win!",
                             "draw": f"The tower of {tower.MAX_PIECES} stands: a draw."}[g["result"]]
                g["phase"] = "over"
            elif who == "human":
                tower_robot(i)
            else:
                g["phase"], g["note"] = "human", "Your move: choose how far off-centre to put your tile, then press Place."

    # ---------------- web server (thread) ----------------
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    web = FastAPI()
    waiting = []

    @web.get("/health")
    def health():   # 15 Oct: the site page (projects/play) checks this before it embeds the game; any origin may ask
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": True}, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})

    @web.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @web.get("/playground.js")
    def js():
        return FileResponse(STATIC / "playground.js", media_type="text/javascript")

    if Path(args.site).is_dir():
        web.mount("/site", StaticFiles(directory=args.site, html=True), name="site")

    def status_of(i):
        s = slots[i]
        g = s.g
        chain = s.chain or s.shown
        cmds = [command_prompt(SKILLS[k], o, d) for k, o, d in chain] if chain else []
        st = {"type": "status", "mode": s.mode, "left": round(args.session_s - (time.time() - s.claimed)), "note": s.note,
              "running": s.chain is not None, "commands": cmds, "marked": s.stage, "done_p": round(s.done_p, 2), "reached": s.reached,
              "backend": args.backend}
        if s.mode == "ttt":
            st["game"] = {"board": g.get("board"), "turn": g.get("turn"), "result": g.get("result"), "note": g.get("note"), "diff": g.get("diff")}
        elif s.mode == "tower":
            st["game"] = {"pieces": [{"owner": p["owner"], "type": p["type"]} for p in g["tw"].pieces], "turn": g["phase"], "result": g["result"],
                          "note": g["note"], "max": tower.MAX_PIECES, "max_offset_cm": tower.MAX_OFFSET * 100}
        return st

    @web.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        sid = os.urandom(8).hex()
        mine = None
        with lock:
            waiting.append(sid)
        try:
            while mine is None:
                with lock:
                    free = [i for i, s in enumerate(slots) if s.owner is None]
                    if free and waiting[0] == sid:
                        mine = free[0]
                        waiting.pop(0)
                        slots[mine].owner, slots[mine].claimed = sid, time.time()
                        slots[mine].reached, slots[mine].stage, slots[mine].done_p, slots[mine].chain = False, 0, 0.0, None
                    pos = waiting.index(sid) + 1 if sid in waiting else 0
                if mine is None:
                    await sock.send_text(json.dumps({"type": "queue", "position": pos}))
                    await asyncio.sleep(1.0)
            inbox.put(("mode", mine, {"mode": "build"}))
            await sock.send_text(json.dumps({"type": "slot", "session_s": args.session_s, "backend": args.backend, "difficulties": list(ttt.DIFFICULTY)}))
            period = 1.0 / args.fps
            while True:
                try:
                    raw = await asyncio.wait_for(sock.receive_text(), timeout=period)
                    if len(raw) <= 4000:
                        msg = json.loads(raw)
                        with lock:
                            slots[mine].claimed = time.time()          # any action restarts the idle clock
                        t = msg.get("type")
                        if t == "run":
                            chain, why = check_chain(msg.get("chain"))
                            if chain is None:
                                await sock.send_text(json.dumps({"type": "error", "message": why}))
                            else:
                                inbox.put(("run", mine, chain))
                        elif t in ("mode", "new_scene", "move", "place"):
                            inbox.put((t, mine, msg))
                except asyncio.TimeoutError:
                    pass
                with lock:
                    st = status_of(mine)
                    frame = slots[mine].frame
                await sock.send_text(json.dumps(st))
                if frame is not None:
                    await sock.send_bytes(frame)
                if st["left"] <= 0:
                    await sock.send_text(json.dumps({"type": "expired"}))
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            with lock:
                if sid in waiting:
                    waiting.remove(sid)
                if mine is not None and slots[mine].owner == sid:
                    slots[mine].owner, slots[mine].chain = None, None
                    inbox.put(("mode", mine, {"mode": "build"}))

    threading.Thread(target=lambda: uvicorn.run(web, host="127.0.0.1", port=args.port, log_level="warning"), daemon=True).start()
    print(f"playground on http://127.0.0.1:{args.port} ({args.slots} slots, backend {args.backend}, policy port {args.vla_port})", flush=True)

    # ---------------- sim loop (main thread) ----------------
    last_send = 0.0
    while app.is_running():
        while not inbox.empty():
            kind, i, msg = inbox.get()
            with lock:
                s = slots[i]
                try:
                    if kind == "mode":
                        m = msg.get("mode")
                        if m == "ttt":
                            ttt_new(i, msg.get("difficulty"))
                        elif m == "tower":
                            tower_new(i)
                        else:
                            build_new(i)
                    elif kind == "new_scene":
                        {"ttt": lambda: ttt_new(i, s.g.get("diff")), "tower": lambda: tower_new(i)}.get(s.mode, lambda: build_new(i))()
                    elif kind == "run" and s.mode == "build":
                        build_run(i, msg)
                    elif kind == "move" and s.mode == "ttt":
                        ttt_human(i, msg.get("square"))
                    elif kind == "place" and s.mode == "tower":
                        tower_human(i, float(msg.get("dx", 0)), float(msg.get("dy", 0)))
                except Exception as e:                                 # a bad message must not stop the simulator
                    s.note = f"error: {e}"
                    print("playground error:", repr(e), flush=True)
        with lock:
            running = [i for i, s in enumerate(slots) if s.chain is not None and s.settle < 0]
            prompts = {i: marked_sentence([command_prompt(SKILLS[k], o, d) for k, o, d in slots[i].chain], slots[i].stage) for i in running}
        actions = hold.clone()
        if running and args.backend == "teacher":
            a, _ = teacher.act(env)
            actions[running] = a[running].clamp(-1, 1)
        elif running:
            img = env.images()
            st = env.state()
            yaw = env.tcp_yaw()
            state = torch.cat([st["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], st["grip"][:, None] / 0.08], -1)
            ix = torch.tensor(running, device=dev)
            front = img["front"][ix][..., :3].to(torch.uint8).cpu().numpy()
            wrist = img["wrist"][ix][..., :3].to(torch.uint8).cpu().numpy()
            conn.send({"front": front.tobytes(), "wrist": wrist.tobytes(), "hw": front.shape[1:3],
                       "state": state[ix].float().cpu().numpy().tobytes(), "task": [prompts[i] for i in running]})
            r = conn.recv()
            a = torch.from_numpy(np.frombuffer(r["actions"], np.float32).reshape(r["shape"])[:, 0].copy()).to(dev)
            actions[ix] = a[:, :5].clamp(-1, 1)
            with lock:
                for j, i in enumerate(running):
                    s = slots[i]
                    s.done_p = float(a[j, 5])
                    s.streak = s.streak + 1 if s.done_p > DONE_THRESH else 0
                    if s.streak >= DONE_HOLD and s.stage < len(s.chain):
                        s.stage, s.streak = s.stage + 1, 0
        if running:
            ix = torch.tensor(running, device=dev)
            hold[ix, 4] = actions[ix, 4]                          # a stopped slot keeps its fingers as they are
        env.step(actions)
        with lock:
            for i, s in enumerate(slots):
                if s.chain is not None:
                    s.t_run += 1
                    if args.backend == "teacher":
                        s.stage = int(env.completed_final[i])
                    if s.mode == "build":                          # the simulator's own check of the last command's end result, held 0.5 s
                        ok = outcome_now(i, s.chain[-1], s.snap)
                        s.lhold = s.lhold + 1 if ok else 0
                        s.reached |= s.lhold >= 5
                    if s.settle < 0 and s.stage >= len(s.chain):
                        s.settle = 0
                    elif s.settle >= 0:
                        s.settle += 1
                    if s.settle >= SETTLE or s.t_run >= RUN_TIMEOUT:
                        timed_out = s.settle < 0
                        if s.mode == "ttt":
                            ttt_run_end(i, timed_out)
                        elif s.mode == "tower":
                            tower_run_end(i, timed_out)
                        else:                                      # build: the finished sentence and its marks stay on screen
                            s.shown, s.chain = s.chain, None
                elif s.mode == "tower" and s.g.get("phase") in ("settle_h", "settle_r"):
                    tower_tick(i)
        now = time.time()
        if now - last_send >= 1.0 / args.fps:
            last_send = now
            img = env.images()
            fr = img["front"][..., :3].to(torch.uint8).cpu().numpy()
            wr = img["wrist"][..., :3].to(torch.uint8).cpu().numpy()
            with lock:
                for i, s in enumerate(slots):
                    if s.owner is not None:
                        both = np.concatenate([fr[i], wr[i]], 1)[..., ::-1]
                        s.frame = cv2.imencode(".jpg", both, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    sys.exit(0)
