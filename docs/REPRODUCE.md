# Run it

Isaac Lab 2.3.2 / Isaac Sim 5.1, LeRobot 0.6.2, OpenCV with ArUco.

```bash
# clips -> reward template
python -m handtrack.track_objects <clip>
python -m handtrack.make_clips <root> --calib ... --out data/processed/clips_real
python -m motion.keypoint_ref --clips data/processed/clips_real --straight --out data/processed/keypoint_ref_real_straight.npz

# teacher -> recordings -> student -> eval
scripts/run_nolift_pipeline.sh

# eval (student, teacher)
NAME=ours_cyl_marks_nolift G=0 scripts/eval_cliptol_loose.sh
G=0 scripts/eval_teacher_loose.sh
```

| Step | Code |
| :--- | :--- |
| Clip tracking | `handtrack/`, `motion/keypoint_ref.py` |
| Teacher | `sim/envs/keypoint_env.py`, `sim/train_rl.py` |
| Recording | `sim/record_keypoint_rollouts.py` |
| Student | `vla/build_dataset.py`, `scripts/distill_place.sh` |
| Eval | `sim/eval_chains.py` |
