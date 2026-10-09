"""Collect camera frames from PandaTaskEnv observations and write an mp4.

Frames are the front and wrist views side by side, with an optional caption bar.
"""
import textwrap
from pathlib import Path

import cv2
import imageio
import numpy as np


class Recorder:
    def __init__(self, path, env_ids=(0,), fps=10, caption=""):
        self.path = Path(path)
        self.env_ids = list(env_ids)
        self.fps = fps
        self.caption = caption
        self.frames = []

    def add(self, obs, text=""):
        rows = []
        for i in self.env_ids:
            front = obs["front"][i].cpu().numpy()
            wrist = obs["wrist"][i].cpu().numpy()
            rows.append(np.concatenate([front, wrist], axis=1))
        img = np.ascontiguousarray(np.concatenate(rows, axis=0)[..., :3].astype(np.uint8))
        self.frames.append((img, [self.caption] if self.caption else [], text))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # caption bar: the caption and the per-frame text wrapped to the image width (6 Oct, user: show the whole
        # prompt); every frame gets a bar of the same height so the video has one size
        width = self.frames[0][0].shape[1]
        cpl = max(20, int(width / 7.2))
        bars = [(textwrap.wrap(" | ".join(c), cpl) if c else []) + textwrap.wrap(t, cpl)[:7] for _, c, t in self.frames]
        rows = max(len(b) for b in bars)
        frames = []
        for (img, _, _), lines in zip(self.frames, bars):
            if rows:
                bar = np.full((8 * -(-(10 + 16 * rows) // 8), width, 3), 255, np.uint8)
                for j, line in enumerate(lines):
                    cv2.putText(bar, line, (8, 17 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                img = np.concatenate([bar, img], axis=0)
            frames.append(img)
        imageio.mimwrite(self.path, frames, fps=self.fps, codec="libx264", quality=8,
                         macro_block_size=8)
        print(f"wrote {self.path} ({len(frames)} frames)", flush=True)
