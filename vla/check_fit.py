"""Offline sanity check: does a fine-tuned SmolVLA reproduce the actions of its own training data?

    python -m vla.check_fit --policy <pretrained_model dir> --root <lerobot dataset root> [--n 256]

Compares the first predicted action of each chunk with the recorded action, against predicting
the dataset mean. A pipeline bug (normalisation, camera mapping, image format) shows up as an
error no better than the mean.
"""
import argparse

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla.server import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    pol, pre, post = load(args.policy, None, args.device)
    ds = LeRobotDataset(repo_id="local/check", root=args.root)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    pred, true, tasks = [], [], []
    for i in idx:
        f = ds[int(i)]
        batch = {"observation.images.camera1": f["observation.images.front"][None],
                 "observation.images.camera2": f["observation.images.wrist"][None],
                 "observation.state": f["observation.state"][None], "task": [f["task"]]}
        pol.reset()
        with torch.inference_mode():
            chunk = post(pol.predict_action_chunk(pre(batch)))
        pred.append(chunk[0, 0].float().cpu().numpy())
        true.append(f["action"].numpy())
        tasks.append(f["task"])
    pred, true = np.stack(pred), np.stack(true)
    mean = true.mean(0)
    print(f"{len(idx)} frames | per-dim MAE model {np.round(np.abs(pred - true).mean(0), 3)} "
          f"vs mean-predictor {np.round(np.abs(mean - true).mean(0), 3)}")
    for t in sorted(set(tasks)):
        m = np.array([x == t for x in tasks])
        print(f"  {t:40s} n={m.sum():3d}  MAE model {np.abs(pred[m] - true[m]).mean():.3f}  "
              f"mean-predictor {np.abs(mean - true[m]).mean():.3f}")


if __name__ == "__main__":
    main()
