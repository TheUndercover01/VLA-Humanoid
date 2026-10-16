"""Sentence builder in the robot's own scene (15 Oct). The playground (play/playground.py) runs on the game scene of sim/envs/multi_env.py:
yellow supply pads, a board, parked pieces. The final model (ours_cyl_marks_nolift) never saw that scene, and a user's 4-command
cylinder sentence fell apart in it. This server runs the SAME page (play/static/index.html + playground.js, build mode only, ?modes=build)
on the cube + cylinder + target scene of sim/envs/keypoint_env.py that the model was trained and evaluated in.

The player chains commands (the rules of playground.check_chain: a push of the cube only first, the target holds one object, ...); the
robot gets the whole sentence and writes its own "(done)" marks; the page shows them next to the simulator's own check of the last
command's end result (the loose check of the final results, held 0.2 s).

Run (policy server in the lerobot env, this in the Isaac env):  play/run_builder.sh start   -> http://127.0.0.1:8800/?modes=build
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vla_port", type=int, default=6201)
parser.add_argument("--port", type=int, default=8800, help="web server port (127.0.0.1)")
parser.add_argument("--slots", type=int, default=4, help="scenes = players at the same time")
parser.add_argument("--session_s", type=float, default=600.0, help="idle seconds a player keeps a slot")
parser.add_argument("--fps", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import asyncio  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from multiprocessing.connection import Client  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.envs.skill_env import NONE, OTHER, TARGET  # noqa: E402
from vla.prompts import command_prompt, marked_sentence  # noqa: E402

AUTHKEY = b"vla-humanoid"
MAX_CMDS = 10
DONE_THRESH, DONE_HOLD = 0.85, 5           # the eval defaults
SETTLE, HOLD_OK = 15, 2                    # ticks after the last mark before the run ends; ticks the result must hold (loose check)
XY_TOL, Z_TOL, TARGET_Z, LIFT = 0.05, 0.03, 0.06, 0.04
DEST = {0: TARGET, 1: OTHER, 2: NONE}      # the page's destination codes
STATIC = Path(__file__).parent / "static"


def check_chain(chain):
    """playground.check_chain: a push (of the cube) only first, the held object onto the free target or the other object, pick up only
    with an empty hand and nothing on top. Returns (list of (skill, obj, dest) in env codes, None) or (None, why)."""
    if not isinstance(chain, list) or not 1 <= len(chain) <= MAX_CMDS:
        return None, f"1 to {MAX_CMDS} commands"
    pos, held, out = {0: "table", 1: "table"}, None, []
    for i, c in enumerate(chain):
        if not (isinstance(c, list) and len(c) == 3 and c[0] in ("push", "pick-lift", "place-down") and c[1] in (0, 1) and c[2] in (0, 1, 2)):
            return None, f"command {i + 1} is malformed"
        k, o, d = c
        free_target = "target" not in pos.values()
        if k == "push":
            if i or d != 0 or o != 0:
                return None, "a push can only be the first command, and only of the cube"
            pos[o] = "target"
        elif k == "pick-lift":
            if held is not None or d != 2 or pos[1 - o] == "on":
                return None, f"command {i + 1}: cannot pick that up now"
            held = o
        else:
            if held != o or d == 2 or (d == 0 and not free_target and pos[o] != "target"):
                return None, f"command {i + 1}: cannot put that there now"
            pos[o], held = ("target" if d == 0 else "on"), None
        out.append((k, o, DEST[d]))
    return out, None


def end_result(cmd, s, i, start):
    """eval_chains.outcome (loose check) for env i: the command's end result for its object right now."""
    k, obj, dest = cmd
    me, other = (s["blue"][i], s["red"][i]) if obj == 1 else (s["red"][i], s["blue"][i])
    if k == "pick-lift":
        return bool(me[2] - start[2] > LIFT)
    if dest == OTHER:
        return bool(((me[:2] - other[:2]).norm() < XY_TOL) & ((me[2] - other[2] - (tasks.HALF_Z[0] + tasks.HALF_Z[1])).abs() < Z_TOL))
    return bool(((me[:2] - s["target"][i, :2]).norm() < XY_TOL) & (me[2] < TARGET_Z))


class Slot:
    def __init__(self):
        self.owner, self.claimed = None, 0.0
        self.chain = self.shown = None
        self.stage = self.streak = 0
        self.done_p = 0.0
        self.t, self.settle, self.lhold, self.reached = 0, -1, 0, False
        self.start = None
        self.note = ""
        self.frame = None


def main():
    cfg = PandaKeypointEnvCfg(task="c2", cameras=True, image_size=256, terminate_on_success=False, episode_s=1e6, hold_fix=True,
                              action_filter=0.0, max_chain=MAX_CMDS)
    cfg.scene.num_envs = args.slots
    cfg.sim.device = args.device
    env = PandaKeypointEnv(cfg)
    dev, n = env.device, args.slots
    gen = torch.Generator().manual_seed(int(time.time()))
    first = [("pick-lift", 0, NONE)]
    env.set_chain(first)
    env.set_layouts(tasks.sample_layouts("c2", n, gen))
    env.reset()
    _dones = env._get_dones

    def no_reset():                        # the page decides when a scene is reset
        _dones()
        z = torch.zeros(n, dtype=torch.bool, device=dev)
        return z, z.clone()
    env._get_dones = no_reset
    conn = Client(("localhost", args.vla_port), authkey=AUTHKEY)
    slots = [Slot() for _ in range(n)]
    lock, inbox = threading.Lock(), queue.Queue()
    hold = torch.zeros(n, 5, device=dev)
    hold[:, 4] = 1.0

    def reset_scene(i, chain):
        lay = env.forced_layouts.clone()
        lay[i] = tasks.sample_layouts("push" if chain[0][0] == "push" else "c2", 1, gen)[0].to(dev)
        env.set_chain(chain)
        env.set_layouts(lay)
        ids = torch.tensor([i], device=dev)
        env._reset_idx(ids)
        env.episode_length_buf[ids] = 0
        hold[i] = 0.0
        hold[i, 4] = 1.0

    def run(i, chain):
        s = slots[i]
        reset_scene(i, chain)
        st = env.state()
        s.chain, s.shown, s.stage, s.streak, s.done_p = chain, None, 0, 0, 0.0
        s.t, s.settle, s.lhold, s.reached = 0, -1, 0, False
        s.start = (st["blue"] if chain[-1][1] == 1 else st["red"])[i].clone()
        s.note = ""

    def run_ticks(chain):
        return int(10 * (6.0 * len(chain) + 5.0))      # as the evals: 6 s per command + 5 s

    # ---------------- web server (thread) ----------------
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn

    web = FastAPI()
    waiting = []

    @web.get("/health")
    def health():
        return JSONResponse({"ok": True}, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})

    @web.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @web.get("/playground.js")
    def js():
        return FileResponse(STATIC / "playground.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    def status_of(i):
        s = slots[i]
        chain = s.chain or s.shown
        return {"type": "status", "mode": "build", "left": round(args.session_s - (time.time() - s.claimed)), "note": s.note,
                "running": s.chain is not None, "commands": [command_prompt(*c) for c in chain] if chain else [], "marked": s.stage,
                "done_p": round(s.done_p, 2), "reached": s.reached, "backend": "vla"}

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
                        slots[mine].chain = slots[mine].shown = None
                    pos = waiting.index(sid) + 1 if sid in waiting else 0
                if mine is None:
                    await sock.send_text(json.dumps({"type": "queue", "position": pos}))
                    await asyncio.sleep(1.0)
            await sock.send_text(json.dumps({"type": "slot", "session_s": args.session_s, "backend": "vla", "difficulties": []}))
            period = 1.0 / args.fps
            while True:
                try:
                    raw = await asyncio.wait_for(sock.receive_text(), timeout=period)
                    if len(raw) <= 4000:
                        msg = json.loads(raw)
                        with lock:
                            slots[mine].claimed = time.time()
                        t = msg.get("type")
                        if t == "run":
                            chain, why = check_chain(msg.get("chain"))
                            if chain is None:
                                await sock.send_text(json.dumps({"type": "error", "message": why}))
                            else:
                                inbox.put(("run", mine, chain))
                        elif t == "new_scene":
                            inbox.put((t, mine, msg))
                except asyncio.TimeoutError:
                    pass
                with lock:
                    st, frame = status_of(mine), slots[mine].frame
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

    threading.Thread(target=lambda: uvicorn.run(web, host="127.0.0.1", port=args.port, log_level="warning"), daemon=True).start()
    print(f"builder on http://127.0.0.1:{args.port} ({n} slots, policy port {args.vla_port})", flush=True)

    # ---------------- sim loop (main thread) ----------------
    last_send = 0.0
    while app.is_running():
        while not inbox.empty():
            kind, i, msg = inbox.get()
            with lock:
                try:
                    if kind == "run" and slots[i].chain is None:
                        run(i, msg)
                    elif kind == "new_scene" and slots[i].chain is None:
                        reset_scene(i, first)
                        slots[i].shown = None
                except Exception as e:                       # a bad message must not stop the simulator
                    slots[i].note = f"error: {e}"
                    print("builder error:", repr(e), flush=True)
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
                s.lhold = s.lhold + 1 if end_result(s.chain[-1], st, i, s.start) else 0
                s.reached |= s.lhold >= HOLD_OK
                if s.settle < 0 and s.stage >= len(s.chain):
                    s.settle = 0
                elif s.settle >= 0:
                    s.settle += 1
                if s.settle >= SETTLE or s.t >= run_ticks(s.chain):
                    if s.settle < 0:
                        s.note = "Time ran out before the robot marked every command done."
                    print(f"builder slot {i}: {len(s.chain)} commands, marks {s.stage}, reached {s.reached}", flush=True)
                    s.shown, s.chain = s.chain, None
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
