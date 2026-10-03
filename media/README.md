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
| `06_scripted_c3.mp4` | 3 Oct | Scripted expert, C3: push blue onto the target, then stack red on it. Two successful episodes; the expert succeeds on 89/100 (25 s limit). | `sim/eval.py --task c3 --video_envs <ids> ...` |
| `07_raw_rl_c1_it900.mp4` | 3 Oct | Raw-action RL on C1 mid-training (reward fix 1): carries the cube onto the target and never lets go. A "what didn't work" clip. | `sim/eval.py --policy rsl:<ckpt> ...` |
| `08_vocab_action.mp4` | 3 Oct | The vocabulary action: top row z = 0 (each skill's mean motion), bottom row random z. One RL step = one ~2 s motion. | `sim/test_vocab_action.py --video ...` |
| `09_b0_c1.mp4` | 3 Oct | B0, untrained SmolVLA on C1: the arm wanders; 0/100. | `sim/eval.py --policy vla ...` with `vla/server.py --policy base` |
| `10_vocab_rl_c1_wip.mp4` | 3 Oct | Vocabulary RL on C1 mid-training (reward fix 2): reach → pick-lift, then holds the cube up. A "what didn't work" clip. | `sim/eval.py --policy rsl:<ckpt> --vocab ...` |
| `11_vocab_rl_c1_push_vs_lift.mp4` | 3 Oct | Vocabulary RL on C1 (before the lift requirement): top row pushes the cube onto the target, bottom row picks it up and places it. Both counted as success under the old C1 rule. | `sim/eval.py --policy rsl:<ckpt> --vocab ... --video_envs 0,2` |

All commands run as `PYTHONPATH=. <IsaacLab>/isaaclab.sh -p <script> --headless --enable_cameras`. Camera rendering only works on `cuda:0` (Isaac Sim 5.1 limitation); training can use either GPU.
