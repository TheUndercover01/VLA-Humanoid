"""10 Oct diagnostic: random actions (as an untrained policy) in 30 scenes of each kind: how many episodes end in a knock / a moved static piece, and when."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.envs.multi_env import PandaMultiEnv, PandaMultiEnvCfg, board_spec, make_spec, tower_spec  # noqa: E402

rng = np.random.default_rng(3)
K = 30
orig = []
for _ in range(K):
    pos = np.array([[0.0, 0.1], [0.1, -0.1]] + [[0, 0]] * 4, np.float32) + rng.uniform(-0.03, 0.03, (6, 2))
    orig.append(make_spec([(2, 0, 7), (3, 0, 6)], [0, 1], pos))
specs = orig + [board_spec(rng) for _ in range(K)] + [tower_spec(rng) for _ in range(K)]
cfg = PandaMultiEnvCfg(task="c1", ref_path="data/processed/keypoint_ref_real_straight.npz", max_chain=2, seed=0, hold_fix=True, derived=False,
                       action_filter=0.95, terminate_on_success=False, cameras=False, episode_s=20.0, hand_leg=True)
cfg.scene.num_envs = len(specs)
cfg.sim.device = args.device
env = PandaMultiEnv(cfg)
env.set_specs(specs)
env.reset()
n = len(specs)
first_hit = torch.full((n,), -1, device=env.device)
hit_static = torch.zeros(n, dtype=torch.bool, device=env.device)
hit_any = torch.zeros(n, dtype=torch.bool, device=env.device)
for step in range(150):
    a = torch.randn(n, 5, device=env.device).clamp(-1, 1)
    env.step(a)
    new_any = env.episode_done & ~hit_any
    first_hit[new_any] = step
    hit_any |= env.episode_done
    hit_static |= env.last_static_hit & env.episode_done
for name, sl in (("original", slice(0, K)), ("board", slice(K, 2 * K)), ("tower", slice(2 * K, 3 * K))):
    h = hit_any[sl]
    print(f"{name}: episodes ended within 15 s: {int(h.sum())}/{K} (static piece moved in {int(hit_static[sl].sum())}); median step of the end: "
          f"{float(first_hit[sl][h].float().median()) if h.any() else float('nan'):.0f}", flush=True)
env.close()
