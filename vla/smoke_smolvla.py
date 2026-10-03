"""Load lerobot/smolvla_base and time action-chunk inference on a fake observation.

    python -m vla.smoke_smolvla --device cuda:1
"""
import argparse
import time

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(args.device).eval()
    policy.config.device = args.device
    pre, post = make_pre_post_processors(policy.config, dataset_stats=None)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"loaded smolvla_base: {n_params:.0f}M params on {args.device}")

    obs = {k: torch.rand(f.shape) for k, f in policy.config.input_features.items()}
    obs["task"] = "pick up the red cube"

    times = []
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = post(policy.predict_action_chunk(pre(dict(obs))))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"action chunk {tuple(chunk.shape)}, finite={bool(torch.isfinite(chunk).all())}")
    print(f"latency: first {times[0] * 1e3:.0f} ms, median {sorted(times)[len(times) // 2] * 1e3:.0f} ms")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
