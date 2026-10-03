"""Generate the fixed eval start states: 100 layouts per task, seed 1234. Run once.

    python -m sim.make_eval_states
"""
from pathlib import Path

import torch

from sim.envs.tasks import TASKS, sample_layouts

N, SEED = 100, 1234
OUT = Path(__file__).parent / "eval_states"


def main():
    OUT.mkdir(exist_ok=True)
    for i, task in enumerate(TASKS):
        gen = torch.Generator().manual_seed(SEED + i)
        layouts = sample_layouts(task, N, gen)
        torch.save({"task": task, "seed": SEED + i, "layouts": layouts}, OUT / f"{task}.pt")
        print(task, tuple(layouts.shape), layouts.min(0).values.tolist(), layouts.max(0).values.tolist())


if __name__ == "__main__":
    main()
