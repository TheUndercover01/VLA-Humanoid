#!/bin/bash
# 9 Oct 23:15: the one chain not run for ours_cyl_marks_realref (c3: push the cylinder to the target, then stack the cube on it), GPU 0, 100 layouts.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=ours_cyl_marks_realref; GPU=0; PORT=6193
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_${NAME}_c3.log 2>&1) &
S=$!
until grep -q "serving" $C/server_${NAME}_c3.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 1200 scripts/run_isaac.sh $C/ev_${NAME}_c3.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
  --hold_fix --chain c3 --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --name $NAME --out runs/eval/${NAME}_c3.csv
pkill -9 -f "sim/eval_chains.py.*runs/eval/${NAME}_c3.csv" 2>/dev/null
echo "$(date '+%a %H:%M') REALREF $NAME c3: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_${NAME}_c3.log | tail -1)" | tee -a $STATUS
kill $S
