#!/bin/bash
# Early look: unstack, restack, restack8 for ours_cyl_marks_pause on the idle GPU 1, parallel to the pipeline's own eval run.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; NAME=ours_cyl_marks_pause; PT=0.0134; PORT=6290
log() { echo "$(date '+%a %H:%M') PAUSE-FIRST $*" | tee -a $C/distill_status.txt; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_pause_first.log 2>&1) &
S=$!
until grep -q "serving" $C/server_pause_first.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for ch in unstack restack restack8; do
  CUDA_VISIBLE_DEVICES=1 timeout 3000 scripts/run_isaac.sh $C/ev_pause_first_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --stack_ztol $PT \
    --name pause_first_$ch --out runs/eval/pause_first_$ch.csv
  log "$ch: $(grep -aoE 'lenient [0-9]+/100, in-order [0-9]+/100 \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_pause_first_$ch.log | tail -1)  [cliptol: unstack 96/51, restack 10/7, restack8 97/0]"
done
kill $S
