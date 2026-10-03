# Progress videos

Rendered in Isaac Sim. Left: front camera, right: wrist camera. Each clip shows two eval start states (rows).
The mp4 files are not committed (`*.mp4` is gitignored); regenerate them with the commands below.

| File | Date | What it shows | Command |
|---|---|---|---|
| `01_reach_probe.mp4` | 3 Oct | First render of the scene (old lighting). TCP driven to x = 0.20 m (reachable) and 0.45 m (beyond reach): the reach measurement that set the workspace. | `sim/probe_reach.py --video ...` |
| `02_scripted_lift.mp4` | 3 Oct | Scripted expert, lift. 100/100 on the eval states. | `sim/eval.py --task lift --policy scripted --video ...` |
| `03_scripted_push.mp4` | 3 Oct | Scripted expert, push: two-fingertip closed-loop push. 100/100 headless. | `sim/eval.py --task push ...` |
| `04_scripted_c1.mp4` | 3 Oct | Scripted expert, C1 pick and place on the target. 100/100. | `sim/eval.py --task c1 ...` |
| `05_scripted_c2.mp4` | 3 Oct | Scripted expert, C2 stack red on blue. 100/100. | `sim/eval.py --task c2 ...` |
| `06_scripted_c3.mp4` | 3 Oct | Scripted expert, C3: push blue onto the target, then stack red on it. Two successful episodes; the expert succeeds on 67-68/100 (C3 is parked). | `sim/eval.py --task c3 --video_envs <ids> ...` |

All commands run as `PYTHONPATH=. <IsaacLab>/isaaclab.sh -p <script> --headless --enable_cameras`. Camera rendering only works on `cuda:0` (Isaac Sim 5.1 limitation); training can use either GPU.
