#!/bin/bash
# 10 Oct (user: ONE teacher for every game): the privileged RL teacher on a mixture of scenes (sim/envs/multi_env.py, VLA_GAME=mix):
# 30% the original cube + cylinder scenes, 35% tic-tac-toe boards (target pad on a free square of a 3x3 board, 0-7 static pieces, the piece to
# move at the supply pad), 35% towers of 1-4 alternating disc / tile pieces (the piece to move goes on top). Same reward as the headline
# (clip-derived keypoint templates + the hand's leg, no scripted states), at most 2 commands per episode, action low-pass 0.95, 20 s episodes,
# a static piece moved > 2 cm ends the episode with the knock penalty.
#   GPU=1 ENVS=8192 ITERS=1000 setsid nohup scripts/train_game_teacher.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=multi VLA_GAME=mix VLA_TRAIN_TYPES="0,1,3,4" VLA_HOLDOUT=1 VLA_FIXED_TARGET="0.03,-0.12" VLA_PUSH_C3=0.5
RUN=${RUN:-game_mix_s1}; GPU=${GPU:-1}; ENVS=${ENVS:-8192}; ITERS=${ITERS:-1000}
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
echo "$(date '+%a %H:%M') GAME teacher $RUN started ($ENVS envs, $ITERS iterations, GPU $GPU)" | tee -a $STATUS
RES=$(ls runs/rl/$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)
RES=${RES:-$INIT}                          # INIT=runs/rl/game_mix_s2/model_999.pt: start a new run from that run's weights (11 Oct: fine-tune on the revised scenes)
CUDA_VISIBLE_DEVICES=$GPU scripts/run_isaac.sh $C/train_$RUN.log sim/train_rl.py --headless --device cuda:0 --task keypoints --seed 1 \
  --num_envs $ENVS --max_iterations $ITERS --run_name $RUN --max_chain 2 --hold_fix --action_filter 0.95 --episode_s 20 --save_interval 20 \
  --knock_penalty --hand_leg --bank_p 0 --ref data/processed/keypoint_ref_real_straight.npz --bank data/processed/state_bank_standin.npz ${RES:+--resume $RES}
echo "$(date '+%a %H:%M') GAME teacher $RUN ended: $(grep -a 'Learning iteration' $C/train_$RUN.log | tail -1 | tr -d '\033' | cut -c1-80)" | tee -a $STATUS
