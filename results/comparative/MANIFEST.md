# Manifest of large artifacts (archive: /media/storage/ayush/archive_comparative_7oct/)

* `models/<student>/`: `checkpoints/last/pretrained_model` of every SmolVLA / X-VLA / DiT student (about 870 MB each, X-VLA 1.7 GB). Originals: `/media/storage/ayush/vla_data/train/<student>/`. Paired 32K snapshot: `ours_kp_filt095_paired_ck32k`.
* `teachers/<run>/`: `model_900.pt` of the 0.85 teacher, `saturated_260.pt` of the 0.95 teacher, `model_240.pt` of the first cube + cylinder teacher, `final_teacher.pt` (= model_399) of the fixed-target teacher. Originals: `runs/rl/<run>/`.
* `datasets/`: the LeRobot datasets of `ours_kp_filt095_paired`, `ours_cyl_fixed_done` (one command + done channel), `ours_cyl_memory` (whole sentence + done count). Originals: `/media/storage/ayush/vla_data/lerobot/`.
* `videos/`: every video of `media/` (runs of teachers and students; `70_`-`76_` are the done-flag and memory videos of 6-7 Oct).
* Not archived (too large, regenerable): the raw rollouts with images, `data/processed/rollouts/` (3000 episodes per student; recorded by `sim/record_keypoint_rollouts.py`), the 12 build shards, intermediate checkpoints every 2000 steps.

## Teachers
| run | environment | what is special | result |
|---|---|---|---|
| `keypoints_hold_filt085_ep20_chain2_s1` | red/blue cubes | action filter 0.85, hold fix, 20 s episodes, stand-in reference | basis of ours_kp_filt085_seq |
| `keypoints_hold_filt095_ep20_chain2_s1` | red/blue cubes | filter 0.95, `saturated_260` (99% skills, 97% chains in training) | basis of the paired students |
| `keypoints_cyl_straight_filt095_ep20_chain2_s1` | cube + cylinder, random target | real straight reference; stopped at iteration ~410; c3-style pushes added from iteration 244 | best c3 42 lenient at checkpoint 280; never passed the 97 gate |
| `keypoints_cyl_fixedtgt_c3half_filt095_s1` | cube + cylinder, FIXED target | 50% c3-style pushes, 400 iterations, real straight reference | model_399: see COMPARISON.md group C |

## Students (see `students.csv` for the details)
`ours_kp_filt085_seq`, `ours_kp_filt095sat_seq`, `ours_kp_filt085_hist`, `ours_kp_v2`, `ours_kp_filt085_seq_xvla`, `ours_kp_filt095_paired` (+ `_ck32k`), `ours_kp_filt095_done_scratch` / `_from48k` (the other experiment), `ours_cyl_fixed_done`, `ours_cyl_memory`; early baselines `b1_standin`, `b2_dart`, `oracle_*`, `ours_dart`, `ours_kp_hold_seq*`, `ours_kp_filt07_seq*` (PROGRESS.md).
