"""Watchdog for a training run that was started WITH it (6 Oct, user: stop training that is not going anywhere).

    python scripts/early_stop.py --log <train log> --pid <trainer pid> [--window 15000] [--min_gain 0.01] [--hang_min 20]

Reads the lerobot log (step:N ... loss:X every 100 steps). It SIGTERMs the trainer when
  * the log has not grown for --hang_min minutes (hung / crashed GPU), or
  * the loss has stalled: the mean loss over the last --window steps is less than --min_gain (relative) below the
    mean over the window before it (only checked once 2 windows exist), or
  * the loss is NaN.
Checkpoints are saved every 2000 steps, so stopping keeps the last one. Writes the reason to <log>.early_stop and exits
when the trainer ends on its own. Only ever given the PID of a run launched with it; never attached to someone else's job.
"""
import argparse
import os
import re
import signal
import time

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--pid", type=int, required=True)
ap.add_argument("--window", type=int, default=15000)
ap.add_argument("--min_gain", type=float, default=0.01)
ap.add_argument("--hang_min", type=float, default=20)
ap.add_argument("--poll_s", type=float, default=120)
a = ap.parse_args()
pat = re.compile(r"step:(\d+)(K?) .*?loss:([\d.eE+-]+|nan)")


def alive():
    try:
        os.kill(a.pid, 0)
        return True
    except OSError:
        return False


def losses():
    out = {}
    with open(a.log, errors="ignore") as f:
        for line in f.read().replace("\r", "\n").splitlines():
            m = pat.search(line)
            if m:
                step = int(m.group(1)) * (1000 if m.group(2) else 1)
                out[step] = float(m.group(3))
    return out


def stop(why):
    msg = f"{time.strftime('%a %H:%M')} early stop: {why}"
    print(msg, flush=True)
    open(a.log + ".early_stop", "w").write(msg + "\n")
    os.kill(a.pid, signal.SIGTERM)


while alive():
    time.sleep(a.poll_s)
    if not os.path.exists(a.log):
        continue
    if time.time() - os.path.getmtime(a.log) > a.hang_min * 60:
        stop(f"log silent for {a.hang_min:.0f} min")
        break
    L = losses()
    if not L:
        continue
    steps = sorted(L)
    last = steps[-1]
    if L[last] != L[last]:
        stop(f"loss is NaN at step {last}")
        break
    if last >= 2 * a.window:
        cur = [L[s] for s in steps if s > last - a.window]
        prev = [L[s] for s in steps if last - 2 * a.window < s <= last - a.window]
        c, p = sum(cur) / len(cur), sum(prev) / len(prev)
        if (p - c) / p < a.min_gain:
            stop(f"loss stalled at step {last}: {p:.4f} -> {c:.4f} over {a.window} steps (< {a.min_gain:.0%} gain)")
            break
