#!/bin/bash
# 14 Oct (user): the cliptol student with a looser bar on every check: xy 5 cm (was 3), on-the-other-object height +-3 cm (1.34),
# on-target height < 6 cm (4.5), pick-up rise 4 cm (5), result held 0.2 s (0.5) for both lenient and in-order. Robot unchanged
# (done flag 0.85 / 5 ticks). GPU 0, next to the pause pipeline.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; NAME=${NAME:-ours_cyl_marks_cliptol}; PORT=${PORT:-6292}; G=${G:-0}; TAG=${TAG:-${NAME#ours_cyl_marks_}}
LOOSE="--xy_tol 0.05 --stack_ztol 0.03 --target_z 0.06 --lift_rise 0.04 --lenient_hold 2 --ordered_hold 2"
log() { echo "$(date '+%a %H:%M') LOOSE $NAME $*" | tee -a $C/distill_status.txt; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_${TAG}_loose.log 2>&1) &
S=$!
until grep -q "serving" $C/server_${TAG}_loose.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for ch in ${CHAINS:-unstack restack restack8 rand8_1 rand8_2 c3 lift c1 c2 swap push lift_cyl c1_cyl}; do
  L=$C/ev_${TAG}_loose_$ch.log
  CUDA_VISIBLE_DEVICES=$G timeout 3000 scripts/run_isaac.sh $L sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt $LOOSE \
    --name ${TAG}_loose_$ch --out runs/eval/${TAG}_loose_$ch.csv
  log "$ch: $(grep -aoE 'lenient [0-9]+/100, in-order [0-9]+/100 \(mean commands in order [0-9.]+/[0-9]+\)' $L | tail -1)"
done
kill $S
