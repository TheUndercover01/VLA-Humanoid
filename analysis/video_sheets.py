"""One contact sheet (8 frames) per video, to check videos by eye: python3 analysis/video_sheets.py media/objects/*.mp4 --out DIR"""
import argparse
from pathlib import Path

import imageio
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)
for v in a.videos:
    frames = [f for f in imageio.get_reader(v)]
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    sel = [frames[i][::2, ::2] for i in idx]
    imageio.imwrite(Path(a.out) / (Path(v).stem + ".png"), np.concatenate([np.concatenate(sel[:4], 1), np.concatenate(sel[4:], 1)], 0))
print(f"{len(a.videos)} sheets in {a.out}")
