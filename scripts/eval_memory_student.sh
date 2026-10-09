#!/bin/bash
# Evaluate a whole-prompt + done-memory student (build_dataset --prompt_mode memory): every step it gets the WHOLE chain sentence and its
# own done-flag count; its flag (done_thresh 0.85, 5 ticks, the eval_chains defaults) raises the count. Lenient + in-order per chain.
#   NAME=ours_cyl_memory PORT=6111 GPU=1 [CK=last] scripts/eval_memory_student.sh
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?}; PORT=${PORT:-6111}; GPU=${GPU:-1}; CK=${CK:-last}; FLAGS=${FLAGS:---memory_done};   # FLAGS=--marks for a build_dataset --prompt_mode marks student
 CHAINS=${CHAINS:-"push lift lift_cyl c1 c1_cyl c2 c3 swap unstack restack"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
[ -n "$NOWAIT" ] || until grep -aq "^EXIT" $C/ft_$NAME.log 2>/dev/null; do sleep 60; done   # NOWAIT=1: evaluate a checkpoint of a run that is still training
log "fine-tuned $NAME: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/$CK/pretrained_model --port $PORT > $C/server_${NAME}_$CK.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_${NAME}_$CK.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$NAME: server failed, see $C/server_${NAME}_$CK.log"; exit 1; }
for ch in $CHAINS; do
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout 1800 scripts/run_isaac.sh $C/mem_${NAME}_${CK}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done $FLAGS --whole_prompt \
    --name ${NAME}_mem --out runs/eval/${NAME}_${CK}_mem_$ch\_val.csv
  flock -u 9; exec 9>&-
  log "VLA $NAME [WHOLE prompt + done memory$FLAGS, $CK] $ch: $(grep -aoE 'lenient [0-9]+/100, in-order [0-9]+/100 \(mean commands in order [0-9.]+/[0-9]+\)' $C/mem_${NAME}_${CK}_$ch.log | tail -1)"
done
kill $SERVER
log "$NAME [WHOLE prompt + done memory$FLAGS, $CK] eval done"
