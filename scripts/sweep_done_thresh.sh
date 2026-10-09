#!/bin/bash
# Done-flag threshold sweep on an existing done-token student (eval only, 7 Oct): how the in-order score of the long chains moves with
# the flag threshold and the number of consecutive ticks it must stay above it (eval_chains.py --done_thresh / --done_hold).
#   NAME=ours_cyl_fixed_done GPU=1 PORT=6106 setsid nohup scripts/sweep_done_thresh.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?train name}; GPU=${GPU:-1}; PORT=${PORT:-6106}; CHAINS=${CHAINS:-"unstack restack c3 swap"}
SETTINGS=${SETTINGS:-"0.7:5 0.85:8"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_${NAME}_sweep.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_${NAME}_sweep.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$NAME sweep: server failed"; exit 1; }
for s in $SETTINGS; do
  th=${s%%:*}; hold=${s##*:}
  for ch in $CHAINS; do
    exec 9>$C/isaac_eval.lock; flock 9
    CUDA_VISIBLE_DEVICES=0 timeout 1800 scripts/run_isaac.sh $C/sweep_${NAME}_${th}_${hold}_$ch.log sim/eval_chains.py --headless \
      --device cuda:0 --keypoints --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done \
      --done_thresh $th --done_hold $hold --name ${NAME}_sweep --out runs/eval/${NAME}_sweep_${th}_${hold}_$ch.csv
    flock -u 9; exec 9>&-
    log "sweep $NAME done_thresh $th hold $hold $ch: $(grep -aoE 'lenient [0-9]+/100, in-order [0-9]+/100 \(mean commands in order [0-9.]+/[0-9]+\)' $C/sweep_${NAME}_${th}_${hold}_$ch.log | tail -1)"
  done
done
kill $SERVER
log "$NAME done-threshold sweep finished"
