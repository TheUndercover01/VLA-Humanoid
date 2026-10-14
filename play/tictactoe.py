"""Tic-tac-toe against the robot (15 Oct). The robot is the tic-tac-toe student `ours_cyl_marks_ttt` (the whole-sentence student with its own
"(done)" marks, fine-tuned on the 9 board squares as targets, VLA_FIXED_TARGET=board): it only ever hears "pick up the cube, then put the
cube on the target". The game decides WHERE (play/games/ttt.py picks the robot's square; difficulty = how often it plays the perfect move)
and moves the green target pad there. The scene is the one the robot was trained in: one cube, one cylinder, one target pad. The board
(the X's and O's already played) lives on the page, drawn over the 9 squares; the human's O never touches the simulator.

A robot turn: a fresh scene with the pad on the chosen square, the robot runs the sentence; when both commands carry its marks (+1.5 s) or
the time runs out, the simulator checks whether the cube really sits on the square (the loose check of the final results). Yes -> an X;
no -> the robot fumbled and loses its turn.

Run (policy server in the lerobot env, this in the Isaac env, same env vars as the student's training):
  play/run_tictactoe.sh start        -> http://127.0.0.1:8820/
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vla_port", type=int, default=6203)
parser.add_argument("--port", type=int, default=8820, help="web server port (127.0.0.1)")
parser.add_argument("--slots", type=int, default=2, help="scenes = players at the same time")
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--turn_s", type=float, default=25.0, help="seconds the robot has for one move")
parser.add_argument("--session_s", type=float, default=900.0)
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

from play.games import ttt  # noqa: E402
from sim.envs import tasks  # noqa: E402
from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.envs.skill_env import NONE, TARGET  # noqa: E402
from vla.prompts import command_prompt, marked_sentence  # noqa: E402

AUTHKEY = b"vla-humanoid"
R = 0
DONE_THRESH, DONE_HOLD = 0.85, 5          # the eval defaults
SETTLE, HOLD_OK = 15, 2                   # ticks after the last mark before judging; ticks the result must hold (loose check)
XY_TOL, TARGET_Z = 0.05, 0.06             # the loose check of the final results (eval_chains --xy_tol / --target_z)
CHAIN = [("pick-lift", R, NONE), ("place-down", R, TARGET)]
# square index = row * 3 + col as play/games/ttt.py; row = dx (table x), col = dy (table y), as sim/envs/multi_env.SQUARES
SQUARES = np.array([[tasks.BOARD_C[0] + dx * tasks.PITCH, tasks.BOARD_C[1] + dy * tasks.PITCH] for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
                   np.float32)
STATIC = Path(__file__).parent / "static"


class Slot:
    def __init__(self):
        self.owner, self.claimed = None, 0.0
        self.new_game("normal", human_first=True)
        self.frame = None

    def new_game(self, diff, human_first):
        self.board = [None] * 9
        self.diff = diff if diff in ttt.DIFFICULTY else ttt.DEFAULT_DIFFICULTY
        self.phase = "human" if human_first else "robot"
        self.result = None                 # "X" / "O" / "draw"
        self.sq = None                     # the robot's square this turn
        self.chain = None                  # the robot's sentence while it moves, else None
        self.stage = self.streak = 0
        self.done_p = 0.0
        self.t, self.settle, self.ok_hold = 0, -1, 0
        self.note = "Your move: click a square." if human_first else "The robot moves first."


def main():
    cfg = PandaKeypointEnvCfg(task="c1", cameras=True, image_size=256, terminate_on_success=False, episode_s=1e6, hold_fix=True,
                              action_filter=0.0, max_chain=2)
    cfg.scene.num_envs = args.slots
    cfg.sim.device = args.device
    env = PandaKeypointEnv(cfg)
    dev, n = env.device, args.slots
    gen = torch.Generator().manual_seed(int(time.time()))
    env.set_chain(CHAIN)
    env.set_layouts(tasks.sample_layouts("c1", n, gen))
    env.reset()
    _dones = env._get_dones

    def no_reset():                        # the game decides when a scene is reset
        _dones()
        z = torch.zeros(n, dtype=torch.bool, device=dev)
        return z, z.clone()
    env._get_dones = no_reset
    conn = Client(("localhost", args.vla_port), authkey=AUTHKEY)
    slots = [Slot() for _ in range(n)]
    lock, inbox = threading.Lock(), queue.Queue()
    hold = torch.zeros(n, 5, device=dev)
    hold[:, 4] = 1.0

    def layout_for(sq):
        """A c1 layout (as in training: board targets) with the pad moved to square sq and both objects at least 12 cm from it."""
        goal = torch.tensor(SQUARES[sq])
        for _ in range(200):
            lay = tasks.sample_layouts("c1", 1, gen)[0]
            if (lay[0:2] - goal).norm() > 0.12 and (lay[2:4] - goal).norm() > 0.12:
                break
        lay[4:6] = goal
        return lay

    def robot_turn(i):
        s = slots[i]
        s.sq = ttt.choose(list(s.board), "X", ttt.DIFFICULTY[s.diff], random)
        lay = env.forced_layouts.clone()
        lay[i] = layout_for(s.sq).to(dev)
        env.set_chain(CHAIN)
        env.set_layouts(lay)
        ids = torch.tensor([i], device=dev)
        env._reset_idx(ids)
        env.episode_length_buf[ids] = 0
        hold[i] = 0.0
        hold[i, 4] = 1.0
        s.chain, s.stage, s.streak, s.done_p = CHAIN, 0, 0, 0.0
        s.t, s.settle, s.ok_hold = 0, -1, 0
        s.phase = "robot"
        s.note = "The robot is placing its cube on the green square."

    def on_square(st, i):
        """eval_chains.outcome for a put-down on the target (the pad sits on the robot's square), loose check."""
        me = st["red"][i]
        return bool(((me[:2] - st["target"][i, :2]).norm() < XY_TOL) & (me[2] < TARGET_Z))

    def finish(i, s):
        """Game over? Sets result + note; returns True when it is."""
        w = ttt.winner(s.board)
        if w:
            s.result, s.phase = w, "over"
            s.note = {"X": "The robot wins. New game?", "O": "You win! New game?", "draw": "A draw. New game?"}[w]
        return bool(w)

    def judge(i, timed_out):
        s = slots[i]
        placed = s.ok_hold >= HOLD_OK
        if placed:
            s.board[s.sq] = "X"
            s.note = "The robot placed its X. Your move."
        else:
            s.note = ("The robot ran out of time" if timed_out else "The robot thinks it placed the cube, but it is not on the square") + \
                ": it loses this turn. Your move."
        print(f"ttt slot {i}: robot square {s.sq} {'placed' if placed else 'FUMBLED'} ({'timeout' if timed_out else 'marked'}) | "
              f"board {''.join(v or '.' for v in s.board)}", flush=True)
        s.chain = None
        if not finish(i, s):
            s.phase = "human"

    def human_move(i, sq):
        s = slots[i]
        if s.phase != "human" or not isinstance(sq, int) or not 0 <= sq < 9 or s.board[sq]:
            return
        s.board[sq] = "O"
        if not finish(i, s):
            robot_turn(i)

    # ---------------- web server (thread) ----------------
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn

    web = FastAPI()
    waiting = []

    @web.get("/health")
    def health():   # the site page (projects/play) checks this before it embeds the game; any origin may ask
        return JSONResponse({"ok": True}, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})

    @web.get("/")
    def index():
        return FileResponse(STATIC / "tictactoe.html", headers={"Cache-Control": "no-cache"})

    def status_of(i):
        s = slots[i]
        return {"type": "status", "board": s.board, "phase": s.phase, "result": s.result, "diff": s.diff, "sq": s.sq,
                "commands": [command_prompt(*c) for c in s.chain] if s.chain else [], "marked": s.stage, "done_p": round(s.done_p, 2),
                "t": round(s.t / 10, 1), "turn_s": args.turn_s, "note": s.note, "sim_ok": s.ok_hold > 0}

    @web.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        sid, mine = os.urandom(8).hex(), None
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
                        slots[mine].new_game("normal", True)
                    pos = waiting.index(sid) + 1 if sid in waiting else 0
                if mine is None:
                    await sock.send_text(json.dumps({"type": "queue", "position": pos}))
                    await asyncio.sleep(1.0)
            period = 1.0 / args.fps
            while True:
                try:
                    raw = await asyncio.wait_for(sock.receive_text(), timeout=period)
                    if len(raw) <= 1000:
                        msg = json.loads(raw)
                        with lock:
                            slots[mine].claimed = time.time()
                        inbox.put((msg.get("type"), mine, msg))
                except asyncio.TimeoutError:
                    pass
                with lock:
                    st, frame = status_of(mine), slots[mine].frame
                    expired = time.time() - slots[mine].claimed > args.session_s
                await sock.send_text(json.dumps(st))
                if frame is not None:
                    await sock.send_bytes(frame)
                if expired:
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            with lock:
                if sid in waiting:
                    waiting.remove(sid)
                if mine is not None and slots[mine].owner == sid:
                    slots[mine].owner, slots[mine].chain = None, None

    threading.Thread(target=lambda: uvicorn.run(web, host="127.0.0.1", port=args.port, log_level="warning"), daemon=True).start()
    print(f"tictactoe on http://127.0.0.1:{args.port} ({n} slots, policy port {args.vla_port})", flush=True)

    # ---------------- sim loop (main thread) ----------------
    last_send = 0.0
    while app.is_running():
        while not inbox.empty():
            kind, i, msg = inbox.get()
            with lock:
                try:
                    s = slots[i]
                    if kind == "new" and s.chain is None:
                        s.new_game(msg.get("difficulty"), msg.get("first", "human") != "robot")
                        if s.phase == "robot":
                            robot_turn(i)
                    elif kind == "move" and s.chain is None:
                        human_move(i, msg.get("square"))
                except Exception as e:                       # a bad message must not stop the simulator
                    slots[i].note = f"error: {e}"
                    print("ttt error:", repr(e), flush=True)
        with lock:
            running = [i for i, s in enumerate(slots) if s.chain is not None and s.settle < 0]
            prompts = [marked_sentence([command_prompt(*c) for c in slots[i].chain], slots[i].stage) for i in running]
        actions = hold.clone()
        if running:
            img, st = env.images(), env.state()
            yaw = env.tcp_yaw()
            state = torch.cat([st["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], st["grip"][:, None] / 0.08], -1)
            ix = torch.tensor(running, device=dev)
            front = img["front"][ix][..., :3].to(torch.uint8).cpu().numpy()
            wrist = img["wrist"][ix][..., :3].to(torch.uint8).cpu().numpy()
            conn.send({"front": front.tobytes(), "wrist": wrist.tobytes(), "hw": front.shape[1:3],
                       "state": state[ix].float().cpu().numpy().tobytes(), "task": prompts})
            r = conn.recv()
            a = torch.from_numpy(np.frombuffer(r["actions"], np.float32).reshape(r["shape"])[:, 0].copy()).to(dev)
            actions[ix] = a[:, :5].clamp(-1, 1)
            hold[ix, 4] = actions[ix, 4]
            with lock:
                for j, i in enumerate(running):
                    s = slots[i]
                    s.done_p = float(a[j, 5])
                    s.streak = s.streak + 1 if s.done_p > DONE_THRESH else 0
                    if s.streak >= DONE_HOLD and s.stage < len(s.chain):
                        s.stage, s.streak = s.stage + 1, 0
        env.step(actions)
        st = env.state()
        with lock:
            for i, s in enumerate(slots):
                if s.chain is None:
                    continue
                s.t += 1
                s.ok_hold = s.ok_hold + 1 if on_square(st, i) else 0
                if s.settle < 0 and s.stage >= len(s.chain):
                    s.settle = 0
                elif s.settle >= 0:
                    s.settle += 1
                if s.settle >= SETTLE or s.t >= args.turn_s * 10:
                    judge(i, s.settle < 0)
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
