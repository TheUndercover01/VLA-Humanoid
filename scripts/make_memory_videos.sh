#!/bin/bash
# Videos of the whole-prompt + done-memory student on long chains (7 Oct, user: "make vids"): per chain a few layouts, one video each
# (eval_chains.py --video_split), caption = the VLA's done count (its memory), the prompt it gets (the whole sentence), the done flag
# and the commands really done in order. The replay result of every layout goes to the status log.
#   NAME=ours_cyl_memory setsid nohup scripts/make_memory_videos.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?}; GPU=${GPU:-0}; PORT=${PORT:-6114}; TAG=${TAG:-76_cyl_mem}
declare -A ENVS=( [unstack]=0,1,3,2 [restack]=1,3,2,0 [c3]=12,32,1,2 )
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_${NAME}_memvid.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_${NAME}_memvid.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$NAME memory videos: server failed"; exit 1; }
for CHAIN in unstack restack c3; do
  E=${ENVS[$CHAIN]}
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=$GPU timeout 2400 scripts/run_isaac.sh $C/memvid_${NAME}_$CHAIN.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $CHAIN --val --policy vla --vla_port $PORT --replan 1 --learned_done --memory_done --whole_prompt \
    --video media/${TAG}_${CHAIN}.mp4 --video_split --video_envs $E --name ${NAME}_memvid --out runs/eval/${NAME}_memvid_${CHAIN}.csv
  flock -u 9; exec 9>&-
  python3 - <<PY | tee -a $STATUS
import csv, time
rows = {x["state"]: x for x in csv.DictReader(open("runs/eval/${NAME}_memvid_${CHAIN}.csv"))}
for i in "$E".split(","):
    r = rows[i]
    print(time.strftime("%a %H:%M"), f"memory video ${CHAIN} layout {i}: media/${TAG}_${CHAIN}_env{i}.mp4 | commands in order {r['ordered_done']}/{r['n_commands']}, lenient {r['lenient']}, in order {r['ordered']}")
PY
done
kill $SERVER
log "$NAME memory videos finished"
