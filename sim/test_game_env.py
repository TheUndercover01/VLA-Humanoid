"""10 Oct smoke test of the game scenes (VLA_OBJECTS=multi VLA_GAME=mix): build board / tower / original scenes, let them settle with zero
actions, check every piece sits where the spec says, nothing moved (no static_hit, no knock), the observation size, and save front-camera frames."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/game_frames")
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.multi_env import PandaMultiEnv, PandaMultiEnvCfg, board_spec, tower_spec  # noqa: E402

rng = np.random.default_rng(1)
specs = [board_spec(rng, n_stat=0), board_spec(rng, n_stat=4), board_spec(rng, n_stat=7, other_on=False), board_spec(rng, n_stat=6, other_on=True),
         tower_spec(rng, h=1), tower_spec(rng, h=2), tower_spec(rng, h=4), tower_spec(rng, h=6)]
n = len(specs)
cfg = PandaMultiEnvCfg(task="c1", ref_path="data/processed/keypoint_ref_real_straight.npz", max_chain=2, seed=0, hold_fix=True, derived=False,
                       action_filter=0.95, terminate_on_success=False, cameras=True, image_size=256, episode_s=20.0, hand_leg=True)
cfg.scene.num_envs = n
cfg.sim.device = args.device
env = PandaMultiEnv(cfg)
env.set_specs(specs)
obs, _ = env.reset()
print("obs dim", obs["policy"].shape, "cfg", cfg.observation_space, "statics", env.n_static, flush=True)
zero = torch.zeros(n, 5, device=env.device)
zero[:, 4] = 1.0
for _ in range(args.steps):
    obs, *_ = env.step(zero)
P, S = env.obj_pos(), env.static_pos()
bad = 0
for i, sp in enumerate(specs):
    for t in range(6):
        if sp["active"][t]:
            xy = P[i, t, :2].cpu().numpy()
            if np.linalg.norm(xy - sp["pos"][t]) > 0.02:
                bad += 1
                print(f"env {i}: type {t} at {xy.round(3)} expected {sp['pos'][t].round(3)}")
    for q, xy, z in sp["statics"]:
        d = np.linalg.norm(S[i, q, :2].cpu().numpy() - xy)
        if d > 0.02 or abs(float(S[i, q, 2]) - z) > 0.01:
            bad += 1
            print(f"env {i}: static {q} at {S[i, q].cpu().numpy().round(3)} expected {np.append(xy, z).round(3)}")
    tgt = env.tgt_now[i].cpu().numpy()
    if np.linalg.norm(tgt - sp["tgt"]) > 1e-4:
        bad += 1
        print(f"env {i}: target {tgt} expected {sp['tgt']}")
print("mismatches:", bad, "| static_hit:", env.static_hit.cpu().tolist(), "| knocked:", env.knocked.cpu().tolist(), flush=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
img = env.images()
for i in range(n):
    f = img["front"][i, ..., :3].to(torch.uint8).cpu().numpy()[..., ::-1]
    w = img["wrist"][i, ..., :3].to(torch.uint8).cpu().numpy()[..., ::-1]
    cv2.imwrite(str(out / f"scene_{i}.jpg"), np.concatenate([f, w], 1))
env.close()
