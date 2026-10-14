"""13 Oct end-to-end test of play/playground.py through its websocket (isaaclab env, has `websockets`): the sentence builder, a tic-tac-toe game
played to the end (the human moves on free squares in order), and a tower duel of up to 4 turns.   python play/test_playground.py [port]"""
import asyncio
import json
import sys

import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8800


async def status(ws, until, timeout=400):
    """Read frames until until(status) is true; returns the last status."""
    async def loop():
        frames, st = 0, None
        while True:
            m = await ws.recv()
            if isinstance(m, str):
                m = json.loads(m)
                if m.get("type") == "status":
                    st = m
                    if until(st):
                        return st, frames
                elif m.get("type") == "error":
                    print("  server error:", m["message"], flush=True)
            else:
                frames += 1
    return await asyncio.wait_for(loop(), timeout)


async def main():
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        # ---- build: "pick up the cube, then put the cube on the target" ----
        await ws.send(json.dumps({"type": "mode", "mode": "build"}))
        await ws.send(json.dumps({"type": "run", "chain": [["pick-lift", 0, 2], ["place-down", 0, 0]]}))
        st, fr = await status(ws, lambda s: s["mode"] == "build" and not s["running"] and s["commands"])
        print(f"build: marks {st['marked']}/{len(st['commands'])}, simulator says reached={st['reached']}, frames {fr}", flush=True)

        # ---- tic-tac-toe ----
        await ws.send(json.dumps({"type": "mode", "mode": "ttt", "difficulty": "normal"}))
        await status(ws, lambda s: s["mode"] == "ttt")
        moves = 0
        while True:
            st, _ = await status(ws, lambda s: s["mode"] == "ttt" and (s["game"]["result"] or (s["game"]["turn"] == "human" and not s["running"])))
            g = st["game"]
            if g["result"]:
                print(f"ttt: result {g['result']} after {moves} human moves | board {g['board']} | {g['note']}", flush=True)
                break
            print(f"  ttt move {moves + 1}: board {g['board']} note '{g['note']}'", flush=True)
            free = [k for k, v in enumerate(g["board"]) if not v]
            await ws.send(json.dumps({"type": "move", "square": free[len(free) // 2]}))
            moves += 1
            await status(ws, lambda s: s["mode"] == "ttt" and (s["game"]["turn"] != "human" or s["game"]["result"]), 60)

        # ---- tower duel ----
        await ws.send(json.dumps({"type": "mode", "mode": "tower"}))
        await status(ws, lambda s: s["mode"] == "tower")
        for turn in range(4):
            st, _ = await status(ws, lambda s: s["mode"] == "tower" and (s["game"]["result"] or (s["game"]["turn"] == "human" and not s["running"])))
            if st["game"]["result"]:
                break
            await ws.send(json.dumps({"type": "place", "dx": 0, "dy": 0}))
            await status(ws, lambda s: s["mode"] == "tower" and s["game"]["turn"] != "human", 60)
        st, _ = await status(ws, lambda s: s["mode"] == "tower" and (s["game"]["result"] or (s["game"]["turn"] == "human" and not s["running"])))
        print(f"tower: {len(st['game']['pieces'])} pieces, result '{st['game']['result']}' | {st['game']['note']}", flush=True)

asyncio.run(main())
