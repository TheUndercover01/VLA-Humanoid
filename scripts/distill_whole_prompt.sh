#!/bin/bash
# Whole-prompt distillation (user, 4 Oct night): a SmolVLA that gets the whole task as one instruction, no
# planner, and has to tell from the images which step it is at. Trained on the keypoint teacher's rollouts
# (at most one hand-off per episode) with sequence labels (vla/build_dataset.py --prompt_mode sequence), then
# evaluated on every keypoint chain twice: whole prompt (no switching) and, for comparison, the same model fed one
# command at a time (planner-style).
#   ROLLOUTS=data/processed/rollouts/ours_kp_hold REC_LOG=.../rec_ours_kp_hold.log NAME=ours_kp_hold_seq FLAGS="--hold_fix" \
#     setsid nohup scripts/distill_whole_prompt.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
ROLLOUTS=${ROLLOUTS:-data/processed/rollouts/ours_kp_hold}
NAME=${NAME:-ours_kp_hold_seq}
FLAGS=${FLAGS:-"--hold_fix"}
C=/media/storage/ayush/cache
D=/media/storage/ayush/vla_data
REC_LOG=${REC_LOG:-$C/rec_ours_kp_hold.log}
STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }

log "whole-prompt distillation $NAME: waiting for the recording ($REC_LOG)"
until grep -aq "successful episodes" $REC_LOG 2>/dev/null; do sleep 60; done
log "recorded: $(grep -aE 'successful episodes' $REC_LOG | tail -1)"

HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD /media/storage/ayush/miniconda3/envs/lerobot/bin/python \
  -m vla.build_dataset --src $ROLLOUTS --repo_id local/$NAME --root $D/lerobot/$NAME --prompt_mode sequence > $C/build_$NAME.log 2>&1
log "dataset (whole-sequence + single-command prompts): $(grep -aE '^wrote' $C/build_$NAME.log | tail -1)"

scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 20000 1 > $C/ft_$NAME.log 2>&1
log "fine-tuned: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"

PORT=6062
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server \
   --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_$NAME.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_$NAME.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
for mode in whole step; do
  extra=""; [ $mode = whole ] && extra="--whole_prompt"
  for ch in push lift c1 c2 c3 swap unstack restack; do
    CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/vla_${NAME}_${mode}_$ch.log sim/eval_chains.py --headless --device cuda:0 \
      --keypoints $FLAGS --chain $ch --val --policy vla --vla_port $PORT --replan 1 $extra --name ${NAME}_$mode \
      --out runs/eval/${NAME}_${mode}_$ch\_val.csv
    log "VLA $NAME [$mode prompt] $ch: $(grep -aoE 'success [0-9]+/100, mean commands done [0-9.]+, failures at .*->' $C/vla_${NAME}_${mode}_$ch.log | sed 's/ ->//')"
  done
done
kill $SERVER
log "whole-prompt distillation done"
