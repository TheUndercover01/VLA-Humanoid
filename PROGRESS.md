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
Not started. See `GPU_START.md`.
