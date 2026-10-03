# Progress

## Fri 2 – Sat 3 Oct (laptop)
Done:
- Repo scaffold, ChArUco board generator (`handtrack/make_board.py`). One board covers camera calibration and the table frame.
- `CLOUD_PROMPT.md`: the full brief (challenge, plan, baselines, protocol, rules).
- **MuJoCo prototype of the task suite** in `sim/`, written but **not yet run or tested**:
  - `scene.py`: Panda + table + red/blue cubes + green target, damped-least-squares Cartesian controller.
  - `env.py`: gymnasium env with tasks `push`, `lift`, `c1`, `c2`, `c3`, success checks, failure-stage tracking, reward, jerk and peak-force metrics.
  - `scripted.py`: privileged-state waypoint expert for every task.
  - `fetch_assets.py`: downloads the Panda model (not committed).

Not done / open:
- The laptop venv (Python 3.11) exists, but the libraries (torch, mujoco, gymnasium, ...) are not installed: the first install failed on a DNS error. Retry `uv pip install`.
- **C3 definition changed.** A "cube in a slot" is physically unsound: any wall that blocks a top-down grasp also blocks the pusher. C3 is now "the cube starts beyond comfortable grasp reach (x in 0.46–0.52 m), push it closer, then pick and place". `C3_FAR_X` in `sim/env.py` is a guess. Measure the real reach limit in sim and set it.
- Eval harness (`sim/eval.py`, saved start states), synthetic vocabulary, VAE, `VocabAction`, RL training, VLA pipeline: not started.

## GPU machine

### Sat 3 Oct: Phase 0 done (gate passed)
Machine: 2× RTX 4090 24 GB (the brief assumed one), driver 535.183.06 (CUDA 12.2), Ubuntu 22.04.4, kernel 6.8, 48 cores, 125 GB RAM. Disk: `/` has only 39 GB free, so envs, caches, Isaac Lab and LeRobot live on `/media/storage` (2.2 TB free).

Envs (recipe in `GPU_START.md` §5): `isaaclab` (Py 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2, torch 2.7.0+cu128, rsl-rl-lib 5.0.1) and `lerobot` (Py 3.12, LeRobot 0.6.2, torch 2.11.0+cu128). Two envs because the Python and numpy requirements conflict.

Lift smoke test (`Isaac-Lift-Cube-Franka-v0`, rsl_rl PPO, 4096 envs, headless), scored by `sim/isaac_check_lift.py` (1024 envs, one 5 s episode):

| Checkpoint | Iterations | Time lifted | Final dist to goal < 5 cm | Median final dist |
|---|---|---|---|---|
| NVIDIA published | — | 0.988 | 1.000 | 3 mm |
| seed 42 (default) | 1500 | 0.032 | 0.000 | 457 mm |
| seed 1 | 3000 | 0.985 | 0.996 | 3 mm |
| seed 2 | 3000 | 0.976 | 0.295 | 65 mm |

1500 iterations take about 13 min on one GPU. Seed 1 matches the published checkpoint, so the install is good. Takeaway for our own RL: the stock lift reward is **seed-sensitive**. Seed 42 never found the lift in 1500 iterations; seed 2 lifts but tracks the goal poorly even at 3000. Plan for several seeds and longer runs than the defaults.

SmolVLA (`vla/smoke_smolvla.py`): `lerobot/smolvla_base` loads (450M params), one 50×6 action chunk takes 198 ms median on a 4090 (first call 619 ms), peak VRAM 0.86 GiB.

Next: Phase 1.1 (task envs in `sim/envs/`), then the eval harness and the scripted expert.
