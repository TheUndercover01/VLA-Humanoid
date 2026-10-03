"""Collect camera frames from PandaTaskEnv observations and write an mp4.

Frames are the front and wrist views side by side, with an optional caption bar.
"""
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
        label = " | ".join(t for t in [self.caption, text] if t)
        if label:
            bar = np.full((32, img.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            img = np.concatenate([bar, img], axis=0)
        self.frames.append(img)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(self.path, self.frames, fps=self.fps, codec="libx264", quality=8,
                         macro_block_size=8)
        print(f"wrote {self.path} ({len(self.frames)} frames)", flush=True)
