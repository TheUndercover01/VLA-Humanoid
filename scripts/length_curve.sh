#!/bin/bash
# 7 Oct evening, "can it remember instructions longer than anything it was trained on?": commands done in order against chain length.
# Random valid chains (sim/eval_chains.py random_chain, eval only): lengths 4 8 12 16 20, seed 1 (the cube is moved first) and seed 0
# (the cylinder), 50 validation layouts each. A shorter chain is a prefix of the longer one with the same seed. With two objects, one
# target and no "put it on the table" command, the long chains repeat the same scenes, so only the policy's memory tells it where it is.
#   NAME=ours_cyl_marks FLAGS=--marks PORT=6112 GPU=0 scripts/length_curve.sh          (a whole-sentence student)
#   NAME=teacher TEACHER=runs/rl/<run>/final_teacher.pt scripts/length_curve.sh         (the RL teacher, privileged state)
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?}; PORT=${PORT:-6111}; GPU=${GPU:-1}; CK=${CK:-last}; FLAGS=${FLAGS:---memory_done}
LENGTHS=${LENGTHS:-"4 8 12 16 20"}; SEEDS=${SEEDS:-"1 0"}; N=${N:-50}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
if [ -n "$TEACHER" ]; then
  POL="rsl:$TEACHER --action_filter 0.95"
else
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
   exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/$CK/pretrained_model --port $PORT > $C/server_len_${NAME}.log 2>&1) &
  SERVER=$!
  until grep -q "serving" $C/server_len_${NAME}.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
  kill -0 $SERVER 2>/dev/null || { log "$NAME: length-curve server failed"; exit 1; }
  POL="vla --vla_port $PORT --replan 1 --learned_done $FLAGS --whole_prompt"
fi
for L in $LENGTHS; do
  for s in $SEEDS; do
    ch=rand${L}_$s
    exec 9>$C/isaac_eval.lock; flock 9
    CUDA_VISIBLE_DEVICES=0 timeout 3600 scripts/run_isaac.sh $C/len_${NAME}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
      --hold_fix --chain $ch --val --num_envs $N --policy $POL --name ${NAME}_len --out runs/eval/len_${NAME}_$ch.csv
    flock -u 9; exec 9>&-
    log "LENGTH $NAME $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/len_${NAME}_$ch.log | tail -1)"
  done
done
[ -n "$SERVER" ] && kill $SERVER
log "LENGTH $NAME curve done"
