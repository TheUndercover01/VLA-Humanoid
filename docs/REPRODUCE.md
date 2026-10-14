# Reproduce it

[← back to the README](../README.md) · [How it works](HOW_IT_WORKS.md) · [Evaluation](EVALUATION.md) · [Results](RESULTS.md) · [Failures](FAILURES.md) · [What didn't work](WHAT_DIDNT_WORK.md)

Environments: Isaac Lab 2.3.2 / Isaac Sim 5.1 (`isaaclab`), LeRobot 0.6.2 (`lerobot`), OpenCV with ArUco (tracking). Two 24 GB GPUs were used; one is enough if the steps run one after the other.

| Step | What happens | Code |
| :--- | :--- | :--- |
| **1. Clips → reward template** | ArUco markers on the cube (ids 1–5) and the can lid (0); the target (11) and a calibration marker (13) give the table plane. A straight line from the clips' mean start offset to the goal, 3 cm tolerance. Lift height, drop height, push direction and end turn come from the clips. | `handtrack/`, `motion/keypoint_ref.py`, `analysis/clip_video.py` |
| **2. Teacher** | PPO, 16,384 envs, 1000 iterations. Object-keypoint reward + the gripper → object leg (`--hand_leg`), random start pose (`--rand_init 1.0`), no scripted states (`--bank_p 0`), no lifting during a push (`--push_lift_pen 0.3`), action low-pass 0.95. | `sim/envs/keypoint_env.py`, `sim/train_rl.py`, `scripts/train_teacher.sh` |
| **3. Recording** | 3000 paired episodes (each scene once per object name) + 1500 push episodes, at most 2 commands each, from random start poses. | `sim/record_keypoint_rollouts.py` |
| **4. Student** | SmolVLA from `smolvla_base`; sentence with its own "(done)" marks, padded to 10 commands; done label on the last 5 frames of each command; early stop (6K-step window, 1.5% gain). Final: `ours_cyl_marks_nolift`, 32K steps. | `vla/build_dataset.py --prompt_mode marks --pad even --max_cmds 10`, `scripts/distill_place.sh` |
| **5. Evaluation** | The sentence is given once; only the policy's own done flag (0.85 for 5 ticks) moves its place. 100 validation layouts per task, [loose check](EVALUATION.md). | `sim/eval_chains.py --learned_done --marks --whole_prompt` |

```bash
# clips -> template
python -m handtrack.cut_session ...; python -m handtrack.track_objects <clip>
python -m handtrack.make_clips <root> --calib ... --out data/processed/clips_real
python -m motion.keypoint_ref --clips data/processed/clips_real --straight --out data/processed/keypoint_ref_real_straight.npz

# teacher -> gate -> recordings -> student -> evals, all in one (detached, logs to the status file)
setsid nohup scripts/run_nolift_pipeline.sh > /dev/null 2>&1 < /dev/null &

# final evaluation with the loose check (student, then teacher)
NAME=ours_cyl_marks_nolift G=0 scripts/eval_cliptol_loose.sh
G=0 scripts/eval_teacher_loose.sh

# videos (16 layouts per task) and the success / failure picks
NAME=ours_cyl_marks_nolift FLAGS="--marks --xy_tol 0.05 --stack_ztol 0.03 --target_z 0.06 --lift_rise 0.04 --lenient_hold 2 --ordered_hold 2" \
  CHAINS="restack unstack restack8" NENV=16 scripts/headline_videos.sh
python3 analysis/pick_videos.py --name ours_cyl_marks_nolift --chains restack unstack restack8 --out media/headline/final --k 2
```

`scripts/eval_cliptol_loose.sh` is named after the first student it scored; `NAME=` picks any student.

Results: per-episode CSVs in `runs/eval/` (`nolift_loose_<task>.csv`, `teacher_loose_nolift_<task>.csv`). The full log with the cause of every number: [PROGRESS.md](../PROGRESS.md).

## Prior work

Learning from Play (Lynch et al. 2019); SPiRL and residual skill policies; LAPA (2410.11758); expert-to-VLA distillation (Exp2VLA 2607.03146, VLA-OPD 2603.26666); RL for VLAs (SimpleVLA-RL 2509.09674); SmolVLA (LeRobot).
