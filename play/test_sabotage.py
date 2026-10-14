"""13 Oct smoke test of play/sabotage.py (run in the isaaclab env, which has `websockets`): plays a few rounds through the websocket
and prints each verdict.  python play/test_sabotage.py [port] [rounds]   Round plan: no sabotage, move at 2 s, fake mark at 2 s, knock at 4 s."""
import asyncio
import json
import sys

import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8810
PLAN = [(None, 0), ("move", 2.0), ("mark", 2.0), ("knock", 4.0)][:int(sys.argv[2]) if len(sys.argv) > 2 else 4]


async def main():
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        frames = 0
        for kind, at in PLAN:
            await ws.send(json.dumps({"type": "start", "task": "c1"}))
            started, fired, last = False, False, None
            while True:
                m = await ws.recv()
                if not isinstance(m, str):
                    frames += 1
                    continue
                m = json.loads(m)
                if m.get("type") != "status":
                    continue
                if m["running"]:
                    started = True
                    if kind and not fired and m["t"] >= at:
                        await ws.send(json.dumps({"type": "sabotage", "kind": kind}))
                        fired = True
                    last = m
                elif started:
                    print(f"sabotage {kind or 'none'} at {at} s | marks {last['marked']}/{len(last['commands'])} | {m['note']} | log {m['log']}", flush=True)
                    break
        print(f"frames received: {frames}; score {m['score']}", flush=True)

asyncio.run(main())
