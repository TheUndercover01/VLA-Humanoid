#!/bin/bash
# 7 Oct evening, "change of mind": a whole-sentence student gets an instruction that is changed mid-run (sim/eval_chains.py EDITS:
# c2_to_target, c1_to_cyl, c2_extend), 100 validation layouts each. Lenient = the NEW instruction's end result; old = the old one's.
#   NAME=ours_cyl_marks FLAGS=--marks PORT=6142 GPU=0 scripts/eval_edits.sh
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?}; PORT=${PORT:-6141}; GPU=${GPU:-1}; CK=${CK:-last}; FLAGS=${FLAGS:---memory_done}; EDITS=${EDITS:-"c2_to_target c1_to_cyl c2_extend"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/$CK/pretrained_model --port $PORT > $C/server_edit_${NAME}.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_edit_${NAME}.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$NAME: edit server failed"; exit 1; }
for e in $EDITS; do
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout 1800 scripts/run_isaac.sh $C/edit_${NAME}_$e.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain edit --edit $e --val --policy vla --vla_port $PORT --replan 1 --learned_done $FLAGS --whole_prompt \
    --name ${NAME}_edit --out runs/eval/edit_${NAME}_$e.csv
  flock -u 9; exec 9>&-
  log "EDIT $NAME $e: new $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+' $C/edit_${NAME}_$e.log | tail -1), old end result $(grep -aoE 'reached [0-9]+/[0-9]+' $C/edit_${NAME}_$e.log | tail -1)"
done
kill $SERVER
log "EDIT $NAME done"
