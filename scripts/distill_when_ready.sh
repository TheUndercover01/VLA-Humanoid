#!/bin/bash
# Distil the keypoint RL teacher into SmolVLA as soon as it is good enough (user, 4 Oct night).
# Every new checkpoint (>= STEP iterations after the last checked one) of the teacher run is scored on
# validation layouts (C1, C2, unstack). When it passes the bar: record successful teacher episodes with
# cameras and DART noise (sim/record_keypoint_rollouts.py), build the LeRobot dataset (per-frame atomic
# instructions), fine-tune SmolVLA 20k steps, evaluate the VLA on every keypoint chain (validation layouts).
#   setsid nohup scripts/distill_when_ready.sh > /dev/null 2>&1 < /dev/null &
# Distil a given checkpoint now, without waiting (the teacher's env flags must match its training):
#   TEACHER=<ckpt> FLAGS="--hold_fix" REC_FLAGS="--hold_fix" NAME=ours_kp_hold setsid nohup scripts/distill_when_ready.sh ...
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/rl/keypoints_hold_v3_chain2_s1}
FLAGS=${FLAGS:-"--hold_fix --release --time_cost 0.05 --knock_penalty --rest_speed 0.05"}
REC_FLAGS=${REC_FLAGS:-"--hold_fix --release --rest_speed 0.05"}
BAR_C1=80; BAR_C2=70; BAR_UNSTACK=60; STEP=300
NAME=${NAME:-ours_kp}
EPISODES=${EPISODES:-1000}
C=/media/storage/ayush/cache
D=/media/storage/ayush/vla_data
STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
latest() { ls $RUN/model_*.pt 2>/dev/null | sort -V | tail -1; }
iter() { basename "$1" .pt | sed 's/model_//'; }
score() {   # checkpoint chain -> success out of 100 on validation layouts
  scripts/run_isaac.sh $C/ready_$2.log sim/eval_chains.py --headless --device cuda:0 --keypoints $FLAGS \
    --chain $2 --val --policy rsl:$1 --name ready --out runs/eval/ready_$2.csv
  grep -aoE "success [0-9]+/100" $C/ready_$2.log | grep -oE "[0-9]+" | head -1
}

if [ -n "$TEACHER" ]; then log "distilling the given teacher $TEACHER ($FLAGS)"; fi
[ -z "$TEACHER" ] && log "watching $RUN (bar: C1 >= $BAR_C1, C2 >= $BAR_C2, unstack >= $BAR_UNSTACK, checked every $STEP iterations)"
last=-$STEP
while [ -z "$TEACHER" ]; do
  ck=$(latest)
  if [ -n "$ck" ] && [ $(iter $ck) -ge $((last + STEP)) ]; then
    last=$(iter $ck)
    c1=$(score $ck c1); c2=$(score $ck c2); un=$(score $ck unstack)
    log "teacher $ck: C1 ${c1:-?} C2 ${c2:-?} unstack ${un:-?}"
    if [ "${c1:-0}" -ge $BAR_C1 ] && [ "${c2:-0}" -ge $BAR_C2 ] && [ "${un:-0}" -ge $BAR_UNSTACK ]; then
      TEACHER=$ck
      break
    fi
  fi
  sleep 600
done

log "teacher $TEACHER: recording $EPISODES episodes"
rm -rf data/processed/rollouts/$NAME
CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/rec_$NAME.log sim/record_keypoint_rollouts.py --headless --device cuda:0 \
  --policy rsl:$TEACHER $REC_FLAGS --episodes $EPISODES --num_envs 96 --noise 0.3 --out data/processed/rollouts/$NAME
log "recorded: $(grep -aE '^kept' $C/rec_$NAME.log | tail -1)"

HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD /media/storage/ayush/miniconda3/envs/lerobot/bin/python \
  -m vla.build_dataset --src data/processed/rollouts/$NAME --repo_id local/$NAME --root $D/lerobot/$NAME > $C/build_$NAME.log 2>&1
log "dataset: $(grep -aE '^wrote' $C/build_$NAME.log | tail -1)"

scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 20000 1 > $C/ft_$NAME.log 2>&1
log "fine-tuned: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"

PORT=6061
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server \
   --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_$NAME.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_$NAME.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
for ch in push lift c1 c2 c3 swap unstack restack; do
  CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/vla_${NAME}_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints $FLAGS --chain $ch --val --policy vla --vla_port $PORT --replan 1 --name $NAME --out runs/eval/${NAME}_$ch\_val.csv
  log "VLA $NAME $ch: $(grep -aoE 'success [0-9]+/100, mean commands done [0-9.]+, failures at .*->' $C/vla_${NAME}_$ch.log | sed 's/ ->//')"
done
kill $SERVER
log "done"
