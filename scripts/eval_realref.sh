#!/bin/bash
# 9 Oct 22:15: evals of ours_cyl_marks_realref (the OLD hand-set-reward teacher, re-recorded with the real-clip template, fine-tuned from
# ours_cyl_marks, 8K steps) on GPU 1 while the derived-reward student trains on GPU 0. Same flags as the headline evals, 100 validation layouts.
#   setsid nohup scripts/eval_realref.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=ours_cyl_marks_realref; GPU=1; PORT=6191
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_${NAME}_g1.log 2>&1) &
S=$!
until grep -q "serving" $C/server_${NAME}_g1.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for ch in unstack restack c2 c1 push lift; do
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 2400 scripts/run_isaac.sh $C/ev_${NAME}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt \
    --name $NAME --out runs/eval/${NAME}_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/${NAME}_$ch.csv" 2>/dev/null
  log "REALREF $NAME $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_${NAME}_$ch.log | tail -1)"
done
kill $S
log "REALREF $NAME eval done"
