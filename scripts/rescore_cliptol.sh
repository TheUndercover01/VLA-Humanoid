#!/bin/bash
# Cliptol student rescored with a looser stack check (teacher's 2.14 cm, then 3 cm), GPU 1.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; NAME=ours_cyl_marks_cliptol; PORT=6291
log() { echo "$(date '+%a %H:%M') RESCORE-CLIPTOL $*" | tee -a $C/distill_status.txt; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_cliptol_rescore.log 2>&1) &
S=$!
until grep -q "serving" $C/server_cliptol_rescore.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for run in "restack 0.0214" "unstack 0.0214" "restack 0.03" "unstack 0.03"; do
  set -- $run; ch=$1; PT=$2; L=$C/ev_cliptol_rescore_${ch}_$PT.log
  CUDA_VISIBLE_DEVICES=1 timeout 3000 scripts/run_isaac.sh $L sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --stack_ztol $PT \
    --name cliptol_rescore_${ch}_$PT --out runs/eval/cliptol_rescore_${ch}_$PT.csv
  log "$ch [stack tolerance $PT m]: $(grep -aoE 'lenient [0-9]+/100, in-order [0-9]+/100 \(mean commands in order [0-9.]+/[0-9]+\)' $L | tail -1)  [at 0.0134: unstack 96/51, restack 10/7]"
done
kill $S
