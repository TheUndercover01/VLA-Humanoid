#!/bin/bash
# 9 Oct, perturbation study on the whole-sentence marks student (eval only, same checkpoint, no training; sim/eval_chains.py --perturb):
# throw the object away after the work is done, drop it mid-carry, move it while the arm reaches, with and without erasing the policy's
# own "(done)" marks at that moment; start with marks already written; ablations: no marks ever (flag ignored), marks written by the sim.
#   MODE=smoke|full setsid nohup scripts/perturb_study.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
MODE=${MODE:-full}; GPU=${GPU:-0}; NAME=${NAME:-ours_cyl_marks}; FLAGS=${FLAGS:---marks}; PORT=${PORT:-6161}
ENVS=$([ $MODE = smoke ] && echo 4 || echo ${ENVS:-50})
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt; V=media/perturb
mkdir -p $V runs/eval/perturb
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_pt_$NAME.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_pt_$NAME.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "PERTURB $NAME: server failed"; exit 1; }
run() {     # tag chain extra-flags
  local tag=$1 ch=$2; shift 2
  local out=runs/eval/perturb/${NAME}_${tag}_$ch$([ $MODE = smoke ] && echo _smoke)
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 3000 scripts/run_isaac.sh $C/pt_${NAME}_${tag}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --num_envs $ENVS --policy vla --vla_port $PORT --replan 1 --learned_done $FLAGS --whole_prompt \
    --name ${NAME}_$tag --out $out.csv --video $V/${NAME}_${tag}_$ch$([ $MODE = smoke ] && echo _smoke).mp4 --video_split --video_envs 0,1,2 "$@"
  flock -u 9; exec 9>&-
  pkill -9 -f "sim/eval_chains.py.*${NAME}_${tag}_$ch" 2>/dev/null   # an Isaac process can hang at shutdown
  log "PERTURB $NAME $tag $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/pt_${NAME}_${tag}_$ch.log | tail -1) | $(grep -aoE 'perturb .*' $C/pt_${NAME}_${tag}_$ch.log | tail -1) | $(grep -aoE 'mean pointer at the end [0-9.]+/[0-9]+' $C/pt_${NAME}_${tag}_$ch.log | tail -1)"
}
if [ $MODE = smoke ]; then
  run knock_um2 c1 --perturb knock --unmark 2
  run drop unstack --perturb drop
  run premark2 unstack --perturb premark --unmark 2
  run oracle unstack --perturb oracle
  kill $SERVER; log "PERTURB $NAME smoke done"; exit 0
fi
# 9 Oct: 5 h left for everything, the core set only (50 scenes each)
run knock c1 --perturb knock
run knock_um2 c1 --perturb knock --unmark 2
run knock_um2 c2 --perturb knock --unmark 2
run drop c1 --perturb drop
run drop_um1 c1 --perturb drop --unmark 1
run move c1 --perturb move
run premark2 unstack --perturb premark --unmark 2
run nomarks unstack --done_thresh 2
run oracle unstack --perturb oracle
run oracle restack --perturb oracle
kill $SERVER
log "PERTURB $NAME done"
