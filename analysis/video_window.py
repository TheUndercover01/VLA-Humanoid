"""14 Oct: a contact sheet of one time window of a video, bigger frames (caption crop kept): python3 analysis/video_window.py VIDEO T0 T1 N OUT.png [cols]
Time in seconds of the video (the eval writes one frame per sim tick of 0.1 s)."""
import sys

import imageio
import numpy as np

v, t0, t1, n, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cols = int(sys.argv[6]) if len(sys.argv) > 6 else 3
r = imageio.get_reader(v)
fps = r.get_meta_data().get("fps", 10)
frames = [f for f in r]
idx = np.linspace(int(t0 * fps), min(int(t1 * fps), len(frames) - 1), n).astype(int)
sel = [frames[i] for i in idx]
while len(sel) % cols:
    sel.append(np.zeros_like(sel[0]))
rows = [np.concatenate(sel[i:i + cols], 1) for i in range(0, len(sel), cols)]
imageio.imwrite(out, np.concatenate(rows, 0))
print(f"{len(frames)} frames at {fps} fps, window {t0}-{t1}s -> {out}")
