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

### Sat 3 Oct: Phase 1.1–1.3 (task envs, eval harness, scripted expert)
Isaac Lab port of the MuJoCo prototype. The MuJoCo files (`sim/env.py`, `scene.py`, `scripted.py`, `fetch_assets.py`) are superseded but kept for reference.
- `sim/envs/panda_env.py`: one DirectRLEnv for all tasks. Panda (high-PD config), table, 5 cm red/blue cubes, visual green target, optional front + wrist cameras. Same 5-D action as the prototype (dx, dy, dz, dyaw, grip), 10 Hz, integrated TCP command tracked by DLS IK at 100 Hz. 21-D state observation in the table frame.
- `sim/envs/tasks.py`: everything task-specific in one module: layout sampling, success, failure-stage tracking, shaped reward.
- `sim/expert.py`: batched scripted expert. `sim/eval.py`: runs any policy on the 100 saved start states per task (`sim/eval_states/*.pt`, seed 1234), writes a CSV per episode (success, failure stage, time, jerk, peak force, skill sentence), optional video. `analysis/results_table.py`: markdown tables with Wilson CIs.

Scripted expert on the eval states (gate "≈100% on C1" passed):

| push | lift | c1 | c2 | c3 |
|---|---|---|---|---|
| 100% | 100% | 100% | 100% | 68% (parked) |

Mean episode time 3.3 s (lift, push), 5.6 s (c1), 5.1 s (c2). Videos: `media/` (see `media/README.md`).

Findings and changes (all feed the README's "what didn't work"):
- **Reach.** Measured with `sim/probe_reach.py`: a vertical gripper reaches ~0.78 m from the base (x = 0.275 at y = 0 in the table frame, 0.25 at y = 0.2) and no closer than ~0.3 m. Tilting the hand gains only ~5 cm. The prototype's ranges (cubes up to x = 0.30, C3 at 0.46–0.52) were out of reach. Objects now spawn in x ∈ [-0.12, 0.18], |y| ≤ 0.2; the TCP command is clamped to 0.32–0.77 m from the base (beyond that the DLS IK settled in a retracted, tilted pose). **Laptop side: retargeted clips should land inside this box when replayed in sim.**
- **C3 redefined again (user decision).** "Push it closer, then pick" is impossible with a vertical gripper: getting behind a cube that is out of grasp reach is even further out. C3 is now "push the blue cube onto the green target, then stack the red cube on it". Its start layouts require the pushing position behind blue to be reachable (0.36–0.74 m from the base); this was added after seeing expert failures on the C3 eval states, before any model was evaluated. **C3 is parked:** the expert gets 68%, all failures are timeouts (long pushes plus pick and place do not fit in 15 s, and pushes from the edge of reach). Likely fixes: a tighter pushability margin and a longer C3 time limit.
- **Pushing.** A single closed-finger contact spun or tipped the cube (4% success). Fixed: hand yawed along the push, fingers at a 3 cm gap so both fingertips touch the face, push at 0.2 m/s, closed loop on the cube's current line to the target, push height 3 cm (2 cm drags the fingertips on the table). Tuned on separate layouts (seed 999), not the eval states. Cube friction 1.5 → 1.0.
- **Slow arm.** The stock high-PD Franka config (damping 80) with the 12 Nm wrist torque limit caps the wrist at 0.15 rad/s, so every Cartesian move crawled at ~6 cm/s. Damping 80 → 30 halved episode times and fixed C1's timeouts.
- **Stock lift reward is seed-sensitive** (Phase 0): plan several seeds for RL.
- Known issue: with cameras on, physics is not bit-identical to headless (push 98% vs 100%). VLA evals run with cameras, so compare models under the same setting.

Next: Phase 1.4, raw-action PPO experts (B2 teachers) on C1 and C2, learning curves against simulated seconds.
