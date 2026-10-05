#!/bin/bash
# Evaluate a fine-tuned VLA (any policy type vla/server.py loads) on every keypoint chain, validation layouts:
# whole prompt (no planner) and one command at a time. Waits for its fine-tune log to finish first.
#   scripts/eval_vla_student.sh <train_name> <port> <server_gpu> [env flags, default --hold_fix]
cd "$(dirname "$0")/.."
NAME=$1; PORT=$2; GPU=$3; FLAGS=${4:-"--hold_fix"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
until grep -aq "^EXIT" $C/ft_$NAME.log 2>/dev/null; do sleep 60; done
log "fine-tuned $NAME: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server \
   --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_$NAME.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_$NAME.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$NAME: server failed, see $C/server_$NAME.log"; exit 1; }
for mode in whole step; do
  extra=""; [ $mode = whole ] && extra="--whole_prompt"
  for ch in push lift c1 c2 c3 swap unstack restack; do
    exec 9>$C/isaac_eval.lock; flock 9            # one Isaac camera eval at a time on GPU 0
    CUDA_VISIBLE_DEVICES=0 timeout 1800 scripts/run_isaac.sh $C/vla_${NAME}_${mode}_$ch.log sim/eval_chains.py --headless \
      --device cuda:0 --keypoints $FLAGS --chain $ch --val --policy vla --vla_port $PORT --replan 1 $extra \
      --name ${NAME}_$mode --out runs/eval/${NAME}_${mode}_$ch\_val.csv
    flock -u 9
    log "VLA $NAME [$mode prompt] $ch: $(grep -aoE 'success [0-9]+/100, mean commands done [0-9.]+, failures at .*->' $C/vla_${NAME}_${mode}_$ch.log | sed 's/ ->//')"
  done
done
kill $SERVER
log "$NAME eval done"
