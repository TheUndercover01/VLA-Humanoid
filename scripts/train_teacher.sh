#!/bin/bash
# The RL teacher of the headline student (run keypoints_cyl_fixedtgt_c3half_filt095_s1, 6 Oct; arguments as in its meta.json):
# cube + cylinder, fixed target, object-keypoint reward from the real clips' straight template, at most 2 commands per episode,
# half of the pushes from the c3 layouts, action low-pass 0.95, 20 s episodes, 400 PPO iterations x 16384 envs.
#   setsid nohup scripts/train_teacher.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
RUN=${RUN:-keypoints_cyl_fixedtgt_c3half_filt095_s1}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU scripts/run_isaac.sh /media/storage/ayush/cache/train_$RUN.log sim/train_rl.py --headless --device cuda:0 \
  --task keypoints --seed 1 --num_envs 16384 --max_iterations 400 --run_name $RUN --max_chain 2 --hold_fix \
  --action_filter 0.95 --episode_s 20 --save_interval 20 \
  --ref data/processed/keypoint_ref_real_straight.npz --bank data/processed/state_bank_standin.npz
