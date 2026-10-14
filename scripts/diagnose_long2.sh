#!/bin/bash
# 13 Oct: follow-up of diagnose_long.sh. The false marks sit on the put-downs (unstack #2 "put the cube on the cylinder" 56% real, restack #2 22%,
# #6 14%) and the flag is not stuck high after a mark (re-arm did nothing), i.e. the flag fires too EARLY. Inference-only test: the flag must hold
# 10 / 15 ticks instead of 5 before a mark is written. unstack + restack, 100 layouts, GPU 1, status lines LONGDIAG2.
cd "$(dirname "$0")/.."
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
until [ $(grep -ac "LONGDIAG .* done (tables" $STATUS) -ge 2 ]; do sleep 30; done
cp $C/longdiag_tables.txt $C/longdiag_tables_full.txt
log() { echo "$(date '+%a %H:%M') LONGDIAG2 $*" | tee -a $STATUS; }
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=ours_cyl_marks_handleg_rand; G=1; PORT=6203; D=/media/storage/ayush/vla_data
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_longdiag2.log 2>&1) &
S=$!
until grep -q "serving" $C/server_longdiag2.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for h in 10 15; do for ch in unstack restack; do
  T=longdiag2_hold${h}_$ch
  CUDA_VISIBLE_DEVICES=$G timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --done_diag --done_hold $h \
    --name $T --out runs/eval/$T.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
  log "$NAME $T: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_$T.log | tail -1) | $(grep -aoE 'done flag audit: [0-9]+ advances, [0-9]+ of them really finished \([0-9]+%\)' $C/ev_$T.log | tail -1)"
done; done
kill $S
log "$NAME done"
