"""Serve SmolVLA action chunks to the Isaac eval harness (run in the lerobot env).

    python -m vla.server --policy /media/storage/ayush/vla_data/train/b1_standin/checkpoints/last/pretrained_model
    python -m vla.server --policy base --stats_from /media/storage/ayush/vla_data/lerobot/b1_standin   # B0

The two sides run in different Python/numpy versions, so requests carry raw bytes:
  {"front", "wrist": uint8 (B, H, W, 3) bytes, "hw": (H, W), "state": float32 (B, 6) bytes, "task": [str] * B}
and the reply is {"actions": float32 (B, chunk, 5) bytes, "shape": (B, chunk, 5)}.
B0 is the base checkpoint with this project's state/action spec and the dataset's normalisation stats.
"""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ADDRESS = ("localhost", 6011)
AUTHKEY = b"vla-humanoid"


def load(policy, stats_from, device):
    if policy == "base":
        pol = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        img = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        pol.config.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                                     "observation.images.camera1": img, "observation.images.camera2": img}
        pol.config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))}
        stats = json.loads((Path(stats_from) / "meta" / "stats.json").read_text())
        rename = {"observation.images.front": "observation.images.camera1",
                  "observation.images.wrist": "observation.images.camera2"}
        stats = {rename.get(k, k): {s: torch.tensor(v) for s, v in d.items()} for k, d in stats.items()}
        pol.config.device = device
        pre, post = make_pre_post_processors(pol.config, dataset_stats=stats)
    else:
        pol = SmolVLAPolicy.from_pretrained(policy)
        pol.config.device = device
        pre, post = make_pre_post_processors(pol.config, pretrained_path=policy)
    return pol.to(device).eval(), pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="pretrained_model dir, or 'base' for B0")
    ap.add_argument("--stats_from", default=None, help="dataset root with meta/stats.json (needed for base)")
    ap.add_argument("--port", type=int, default=ADDRESS[1])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_batch", type=int, default=25, help="envs per forward pass (bounds GPU memory)")
    args = ap.parse_args()
    pol, pre, post = load(args.policy, args.stats_from, args.device)
    with Listener((ADDRESS[0], args.port), authkey=AUTHKEY) as listener:
        print(f"serving {args.policy} on port {args.port}, chunk {pol.config.chunk_size}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    h, w = msg["hw"]
                    imgs = {}
                    for key, cam in [("front", "camera1"), ("wrist", "camera2")]:
                        a = np.frombuffer(msg[key], np.uint8).reshape(-1, h, w, 3)
                        imgs[f"observation.images.{cam}"] = torch.from_numpy(a.copy()).permute(0, 3, 1, 2).float() / 255
                    b = len(msg["task"])
                    batch = {**imgs, "observation.state": torch.from_numpy(
                        np.frombuffer(msg["state"], np.float32).reshape(b, 6).copy()), "task": list(msg["task"])}
                    chunks = []
                    for i in range(0, b, args.max_batch):
                        part = {k: v[i:i + args.max_batch] for k, v in batch.items()}
                        with torch.inference_mode():
                            chunks.append(post(pol.predict_action_chunk(pre(part))).float().cpu())
                    chunk = torch.cat(chunks).numpy().astype(np.float32)
                    conn.send({"actions": chunk.tobytes(), "shape": chunk.shape})


if __name__ == "__main__":
    main()
