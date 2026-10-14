"""Sabotage (13 Oct, user: "player against the robot's reliability ... it forces the player to learn the method to sabotage it perfectly").

Each round the robot (a whole-sentence SmolVLA student with its own "(done)" marks, as in sim/eval_chains.py) gets a short task in the
cube + cylinder scene it was trained in. The player has a small sabotage budget per round:
  move   slide the object the robot is working on to another free spot on the table;
  knock  throw it (a short push in a random direction, as eval_chains --perturb knock);
  mark   write a false "(done)" after the command the robot is on (not the last one): its memory says it is finished with something it is not.
The robot ends a round itself when every command of its sentence carries a mark (then 1.5 s to settle), or the time runs out. The
simulator then checks the real end result (the lenient check of eval_chains): there -> the robot wins the round, not there -> the player.
Best of 5. The page shows the sentence with the robot's marks and its done flag live, i.e. what it believes, next to the camera.

Run (policy server in the lerobot env, this in the Isaac env, same env vars as the student's training):
  play/run_sabotage.sh start [checkpoint name]       -> http://127.0.0.1:8810/
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vla_port", type=int, default=6202)
parser.add_argument("--port", type=int, default=8810, help="web server port (127.0.0.1)")
parser.add_argument("--slots", type=int, default=2, help="scenes = players at the same time")
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--round_s", type=float, default=30.0, help="seconds per round")
parser.add_argument("--budget", type=int, default=1, help="sabotages per round")
parser.add_argument("--session_s", type=float, default=900.0)
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
R, B = 0, 1
DONE_THRESH, DONE_HOLD = 0.85, 5          # the eval defaults
SETTLE, HOLD_OK = 15, 5                   # ticks after the last mark before judging; ticks the end result must hold
ROUNDS = 5
# task name: (commands, layouts drawn as for this eval task, words for the page)
TASKS = {
    "c1": ([("pick-lift", R, NONE), ("place-down", R, TARGET)], "c1", "put the cube on the green target"),
    "c2": ([("pick-lift", R, NONE), ("place-down", R, OTHER)], "c2", "stack the cube on the cylinder"),
    "swap": ([("pick-lift", B, NONE), ("place-down", B, OTHER)], "c2", "stack the cylinder on the cube"),
}
STATIC = Path(__file__).parent / "static"


def end_result(chain, s, i, start):
    """eval_chains.outcome for env i: the last command's end result for its object right now."""
    k, obj, dest = chain[-1]
    me, other = (s["blue"][i], s["red"][i]) if obj == B else (s["red"][i], s["blue"][i])
    if k == "pick-lift":
        return bool(me[2] - start[2] > 0.05)
    if dest == OTHER:
        return bool(((me[:2] - other[:2]).norm() < 0.03) & ((me[2] - other[2] - (tasks.HALF_Z[0] + tasks.HALF_Z[1])).abs() < 0.015))
    return bool(((me[:2] - s["target"][i, :2]).norm() < 0.03) & (me[2] < 0.045))


class Slot:
    def __init__(self):
        self.owner, self.claimed = None, 0.0
        self.task = "c1"
        self.chain = None                  # commands of the round being played, None between rounds
        self.stage = self.streak = 0       # the robot's marks (its own memory) and its done-flag streak
        self.done_p = 0.0
        self.t, self.settle, self.ok_hold = 0, -1, 0
        self.start = None
        self.left = 0                      # sabotages left this round
        self.log = []                      # what the player did this round (shown, and in the server log)
        self.score = {"robot": 0, "player": 0}
        self.round = 0
        self.note = "Pick a task and press Start round."
        self.frame = None


def main():
    cfg = PandaKeypointEnvCfg(task="c2", cameras=True, image_size=256, terminate_on_success=False, episode_s=1e6, hold_fix=True,
                              action_filter=0.0, max_chain=2)
    cfg.scene.num_envs = args.slots
    cfg.sim.device = args.device
    env = PandaKeypointEnv(cfg)
    dev, n = env.device, args.slots
    rng = np.random.default_rng(int(time.time()))
    gen = torch.Generator().manual_seed(int(time.time()))
    env.set_chain(TASKS["c1"][0])
    env.set_layouts(tasks.sample_layouts("c2", n, gen))
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

    def new_round(i, task):
        s = slots[i]
        if task not in TASKS:
            task = "c1"
        if s.round >= ROUNDS:              # a new match
            s.score, s.round = {"robot": 0, "player": 0}, 0
        chain, lay_task, _ = TASKS[task]
        env.set_chain(chain)               # read by the reset of this env only
        lay = env.forced_layouts.clone()
        lay[i] = tasks.sample_layouts(lay_task, 1, gen)[0].to(dev)
        env.set_layouts(lay)
        ids = torch.tensor([i], device=dev)
        env._reset_idx(ids)
        env.episode_length_buf[ids] = 0
        hold[i] = 0.0
        hold[i, 4] = 1.0
        st = env.state()
        s.task, s.chain, s.stage, s.streak, s.done_p = task, chain, 0, 0, 0.0
        s.t, s.settle, s.ok_hold = 0, -1, 0
        s.start = (st["blue"] if chain[-1][1] == B else st["red"])[i].clone()
        s.left, s.log = args.budget, []
        s.round += 1
        s.note = f"Round {s.round} of {ROUNDS}: the robot was told to {TASKS[task][2]}. Sabotage it before it finishes."

    def sabotage(i, kind):
        s = slots[i]
        if s.chain is None or s.left <= 0 or s.settle >= 0 and kind != "mark":
            return
        cmd = s.chain[min(s.stage, len(s.chain) - 1)]
        obj = cmd[1]
        if kind in ("move", "knock"):
            st = env.state()
            keep = [st[k][i, :2].cpu().numpy() for k in ("red", "blue", "target", "tcp")]
            keep.pop(obj)
            lo, hi = np.array(tasks.REGION[0]), np.array(tasks.REGION[1])
            for _ in range(500):
                p = rng.uniform(lo, hi)
                if all(np.linalg.norm(p - q) > 0.10 for q in keep):
                    break
            speed = 0.4 if kind == "knock" else 0.0
            z = tasks.HALF_Z[obj] + (0.03 if speed else 0.002)
            pose = torch.tensor([[p[0], p[1], z, 1, 0, 0, 0]], dtype=torch.float32, device=dev)
            pose[:, :3] += env.table_origin[i:i + 1]
            a = rng.uniform(0, 2 * np.pi)
            vel = torch.tensor([[speed * np.cos(a), speed * np.sin(a), 0, 0, 0, 0]], dtype=torch.float32, device=dev)
            o = env.blue if obj == B else env.red
            o.write_root_pose_to_sim(pose, env_ids=torch.tensor([i], device=dev))
            o.write_root_velocity_to_sim(vel, env_ids=torch.tensor([i], device=dev))
            what = {"move": "moved", "knock": "knocked"}[kind] + f" the {'cylinder' if obj == B else 'cube'}"
        elif kind == "mark":            # never on the last command: that would just end the round (the robot stops at once)
            if s.stage >= len(s.chain) - 1:
                s.note = "A false (done) only works on a command that is not the last one."
                return
            s.stage, s.streak = s.stage + 1, 0
            what = f"wrote a false (done) after \"{command_prompt(*cmd)}\""
        else:
            return
        s.left -= 1
        s.log.append(f"{s.t / 10:.1f} s: you {what}")

    def judge(i, timed_out):
        s = slots[i]
        won = s.ok_hold >= HOLD_OK
        s.score["robot" if won else "player"] += 1
        why = ("it finished the task" if won else
               ("the time ran out" if timed_out else "it believed it was finished, but the task is not done"))
        s.note = (f"Round {s.round}: {'the ROBOT wins' if won else 'YOU win'} ({why}). " +
                  (f"Match over: robot {s.score['robot']}, you {s.score['player']}." if s.round >= ROUNDS else "Press Start round for the next one."))
        print(f"sabotage slot {i} round {s.round} task {s.task}: {'robot' if won else 'player'} | {'; '.join(s.log) or 'no sabotage'}", flush=True)
        s.chain = None

    # ---------------- web server (thread) ----------------
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    import uvicorn

    web = FastAPI()
    waiting = []

    @web.get("/")
    def index():
        return FileResponse(STATIC / "sabotage.html")

    def status_of(i):
        s = slots[i]
        chain = s.chain
        return {"type": "status", "running": chain is not None, "task": s.task, "round": s.round, "rounds": ROUNDS, "score": s.score,
                "commands": [command_prompt(*c) for c in chain] if chain else [], "marked": s.stage, "done_p": round(s.done_p, 2),
                "t": round(s.t / 10, 1), "round_s": args.round_s, "left": s.left, "budget": args.budget, "log": s.log, "note": s.note,
                "sim_ok": s.ok_hold > 0, "tasks": {k: v[2] for k, v in TASKS.items()}}

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
                        slots[mine].score, slots[mine].round, slots[mine].chain = {"robot": 0, "player": 0}, 0, None
                        slots[mine].note = "Pick a task and press Start round."
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
    print(f"sabotage on http://127.0.0.1:{args.port} ({n} slots, policy port {args.vla_port})", flush=True)

    # ---------------- sim loop (main thread) ----------------
    last_send = 0.0
    while app.is_running():
        while not inbox.empty():
            kind, i, msg = inbox.get()
            with lock:
                try:
                    if kind == "start" and slots[i].chain is None:
                        new_round(i, msg.get("task"))
                    elif kind == "sabotage":
                        sabotage(i, msg.get("kind"))
                except Exception as e:                       # a bad message must not stop the simulator
                    slots[i].note = f"error: {e}"
                    print("sabotage error:", repr(e), flush=True)
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
                s.ok_hold = s.ok_hold + 1 if end_result(s.chain, st, i, s.start) else 0
                if s.settle < 0 and s.stage >= len(s.chain):
                    s.settle = 0
                elif s.settle >= 0:
                    s.settle += 1
                if s.settle >= SETTLE or s.t >= args.round_s * 10:
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
