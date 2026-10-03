"""Convert replay/rollout npz files (sim/replay_clips.py and later expert rollouts) into a local LeRobot dataset.

Run in the lerobot env:
    python -m vla.build_dataset --src data/processed/replays/standin --repo_id local/b1_standin \
        --root /media/storage/ayush/vla_data/lerobot/b1_standin
Each npz is one episode: front, wrist (T, H, W, 3) uint8, state (T, 6), action (T, A), prompt.
A = 5 for the raw action (10 fps); vocabulary rollouts (one frame per ~2 s primitive) store
[skill one-hot | z] and use --fps 1 (LeRobot needs an integer rate; only the frame order matters).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "sin_yaw", "cos_yaw", "gripper_gap"]
RAW_ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="one or more folders of npz episodes")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=None)
    args = ap.parse_args()

    files = [f for src in args.src for f in sorted(Path(src).glob("*.npz"))[:args.max_episodes]]   # cap per folder
    first = np.load(files[0])
    a_dim = first["action"].shape[1]
    action_names = RAW_ACTION_NAMES if a_dim == 5 else [f"a{i}" for i in range(a_dim)]
    h, w = first["front"].shape[1:3]
    img = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.images.front": img,
        "observation.images.wrist": img,
        "observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (a_dim,), "names": action_names},
    }
    if Path(args.root).exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(args.repo_id, args.fps, features, root=args.root, robot_type="franka_panda_sim",
                               use_videos=True)
    for f in files:
        ep = np.load(f, allow_pickle=True)
        prompt = str(ep["prompt"])
        for t in range(len(ep["action"])):
            ds.add_frame({"observation.images.front": ep["front"][t], "observation.images.wrist": ep["wrist"][t],
                          "observation.state": ep["state"][t].astype(np.float32),
                          "action": ep["action"][t].astype(np.float32), "task": prompt})
        ds.save_episode()
    ds.finalize()
    print(f"wrote {len(files)} episodes, {ds.num_frames} frames to {args.root}")


if __name__ == "__main__":
    main()
