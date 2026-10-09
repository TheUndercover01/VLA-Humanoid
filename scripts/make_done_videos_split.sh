#!/bin/bash
# Several candidate layouts of one long chain, one video each (7 Oct): the replay result of every layout is printed to the status log
# (commands done in order), so a real success and a real failure can be picked afterwards. Uses the done-flag setting given.
#   NAME=ours_cyl_fixed_done CHAIN=unstack ENVS=1,2,4,5,0,3 TH=0.7 HOLD=5 setsid nohup scripts/make_done_videos_split.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?}; CHAIN=${CHAIN:-unstack}; ENVS=${ENVS:?comma list}; TH=${TH:-0.7}; HOLD=${HOLD:-5}; GPU=${GPU:-0}; PORT=${PORT:-6108}; TAG=${TAG:-75_cyl_done}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_${NAME}_vid2.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_${NAME}_vid2.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$NAME split videos: server failed"; exit 1; }
exec 9>$C/isaac_eval.lock; flock 9
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 scripts/run_isaac.sh $C/vid2_${NAME}_$CHAIN.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
  --hold_fix --chain $CHAIN --val --policy vla --vla_port $PORT --replan 1 --learned_done --done_thresh $TH --done_hold $HOLD \
  --video media/${TAG}_${CHAIN}.mp4 --video_split --video_envs $ENVS --name ${NAME}_vid2 --out runs/eval/${NAME}_vid2_${CHAIN}.csv
flock -u 9; exec 9>&-
python3 - <<PY | tee -a $STATUS
import csv, time
rows = {x["state"]: x for x in csv.DictReader(open("runs/eval/${NAME}_vid2_${CHAIN}.csv"))}
for i in "$ENVS".split(","):
    r = rows[i]
    print(time.strftime("%a %H:%M"), f"video ${CHAIN} layout {i} (done_thresh $TH, hold $HOLD): media/${TAG}_${CHAIN}_env{i}.mp4 | commands in order {r['ordered_done']}/{r['n_commands']}, lenient {r['lenient']}, in order {r['ordered']}")
PY
kill $SERVER
log "$NAME split videos finished"
