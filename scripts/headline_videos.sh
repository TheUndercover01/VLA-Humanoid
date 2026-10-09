#!/bin/bash
# 9 Oct, videos of the headline student (whole sentence, its own "(done)" marks): 8 validation scenes per chain, one video each,
# captions show its pointer and done flag. The VLA samples noise, so check the replays by eye (analysis/video_sheets.py).
#   setsid nohup scripts/headline_videos.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:-ours_cyl_marks}; FLAGS=${FLAGS:---marks}; PORT=${PORT:-6162}; GPU=${GPU:-0}; CHAINS=${CHAINS:-"c2 unstack restack c3 c1"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt; V=media/headline
mkdir -p $V runs/eval/headline
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_hv_$NAME.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_hv_$NAME.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "HEADLINE $NAME: server failed"; exit 1; }
for ch in $CHAINS; do
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 2400 scripts/run_isaac.sh $C/hv_${NAME}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --num_envs 8 --policy vla --vla_port $PORT --replan 1 --learned_done $FLAGS --whole_prompt \
    --name ${NAME}_hv --out runs/eval/headline/${NAME}_$ch.csv --video $V/${NAME}_$ch.mp4 --video_split --video_envs 0,1,2,3,4,5,6,7
  flock -u 9; exec 9>&-
  pkill -9 -f "sim/eval_chains.py.*headline/${NAME}_$ch" 2>/dev/null
  log "HEADLINE $NAME $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/hv_${NAME}_$ch.log | tail -1)"
done
kill $SERVER
log "HEADLINE $NAME done"
