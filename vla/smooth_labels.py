"""Copy a LeRobot dataset with smoothed action labels (the videos are hard-linked, not re-encoded).

The keypoint RL teacher's actions are bang-bang: ~30% of steps flip sign and only 12-28% of their variance
survives a 1 s average, so a VLA cannot tell from two nearly identical frames why the label flipped, and it
regressed to the dataset mean (PROGRESS.md, 5 Oct). Here every frame's movement label (dx, dy, dz, dyaw) becomes
the mean of the teacher's next --window actions in the same episode: where the hand is heading over the next
half second, which the images can explain. The grip label is kept as recorded (a timed open/close, not jittery).
The action statistics in meta/stats.json (used to normalise the actions) are recomputed.

    python -m vla.smooth_labels --src /media/storage/ayush/vla_data/lerobot/ours_kp_hold_seq \
        --dst /media/storage/ayush/vla_data/lerobot/ours_kp_hold_seq_smooth --window 5
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

MOVE = slice(0, 4)          # dx, dy, dz, dyaw; index 4 is the grip


def smooth(actions, window):
    """Forward mean over the next `window` steps (shorter at the episode's end)."""
    out = actions.copy()
    c = np.cumsum(np.vstack([np.zeros((1, 4)), actions[:, MOVE]]), 0)
    n = len(actions)
    for t in range(n):
        e = min(n, t + window)
        out[t, MOVE] = (c[e] - c[t]) / (e - t)
    return out


def stats(x):
    q = {f"q{p:02d}": np.quantile(x, p / 100, axis=0).tolist() for p in (1, 10, 50, 90, 99)}
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)], **q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        shutil.rmtree(dst)
    subprocess.run(["cp", "-al", str(src), str(dst)], check=True)       # hard links: videos are shared
    allact = []
    for f in sorted(dst.glob("data/*/*.parquet")):
        df = pd.read_parquet(f)
        act = np.stack(df["action"].to_numpy()).astype(np.float32)
        new = act.copy()
        for ep in df["episode_index"].unique():
            m = (df["episode_index"] == ep).to_numpy()
            order = np.argsort(df.loc[m, "frame_index"].to_numpy())
            idx = np.nonzero(m)[0][order]
            new[idx] = smooth(act[idx], a.window)
        df["action"] = list(new)
        f.unlink()                                                      # break the hard link before writing
        df.to_parquet(f)
        allact.append(new)
    allact = np.concatenate(allact)
    sp = dst / "meta" / "stats.json"
    st = json.loads(sp.read_text())
    st["action"] = stats(allact)
    sp.unlink()
    sp.write_text(json.dumps(st, indent=2))
    sign = np.sign(allact[:, MOVE])
    print(f"smoothed {len(allact)} frames (window {a.window}) -> {dst}; action std now {np.round(allact.std(0), 2)}; "
          f"fraction at +-1: {np.round((np.abs(allact[:, MOVE]) > 0.98).mean(0), 2)}")


if __name__ == "__main__":
    main()
