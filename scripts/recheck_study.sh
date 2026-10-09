#!/bin/bash
# 8 Oct evening, "check your own work": the marks student re-asks itself, from 1 s after every advance and for 1.5 s, whether the command it
# just marked done is really done (the same VLA, the sentence with that mark removed); if its flag stays below 0.5 for 0.5 s the mark is
# erased (eval_chains --recheck delay,window,low,need). No training, its own outputs only. The done-flag study showed that a stricter flag
# cannot fix the long chains (precision against stalls); this tests whether a CLOSED loop on the marks stops the errors from compounding.
#   MODE=smoke|full setsid nohup scripts/recheck_study.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
MODE=${MODE:-full}; GPU=${GPU:-1}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
serve() {   # name port -> starts the server, sets SERVER
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
   exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$1/checkpoints/last/pretrained_model --port $2 > $C/server_rc_$1.log 2>&1) &
  SERVER=$!
  until grep -q "serving" $C/server_rc_$1.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
  kill -0 $SERVER 2>/dev/null || { log "RECHECK $1: server failed"; exit 1; }
}
run() {     # name port flags tag chain thresh hold vote envs
  local name=$1 port=$2 flags=$3 tag=$4 ch=$5 th=$6 hold=$7 vote=$8 envs=$9
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 2400 scripts/run_isaac.sh $C/rc_${name}_${tag}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --num_envs $envs --policy vla --vla_port $port --replan 1 --learned_done $flags --whole_prompt \
    --done_thresh $th --done_hold $hold --done_vote $vote --done_diag --name ${name}_rc --out runs/eval/rc_${name}_${tag}_$ch.csv
  flock -u 9; exec 9>&-
  pkill -9 -f "sim/eval_chains.py.*rc_${name}_${tag}_$ch" 2>/dev/null   # an Isaac process can hang at shutdown
  log "RECHECK $name $tag $ch (thresh $th, hold $hold, vote $vote): $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/rc_${name}_${tag}_$ch.log | tail -1) | $(grep -aoE 'recheck: [0-9]+ marks erased, [0-9]+ of them rightly' $C/rc_${name}_${tag}_$ch.log | tail -1) | $(grep -aoE 'done flag audit: .*' $C/rc_${name}_${tag}_$ch.log | tail -1)"
}
RC="--marks --recheck 10,15,0.5,5"
if [ $MODE = smoke ]; then
  serve ours_cyl_marks 6161; run ours_cyl_marks 6161 "$RC" smoke rand4_1 0.85 5 1 4; kill $SERVER; exit 0
fi
serve ours_cyl_marks 6161
for ch in rand4_1 rand4_0 rand8_1 unstack restack rand12_1 rand20_1; do
  run ours_cyl_marks 6161 "$RC" rc $ch 0.85 5 1 $([ $ch = unstack -o $ch = restack ] && echo 100 || echo 50)
done
for ch in rand4_1 rand4_0 rand8_1 unstack; do
  run ours_cyl_marks 6161 "$RC" rc_t0.9h8 $ch 0.9 8 1 $([ $ch = unstack ] && echo 100 || echo 50)
done
kill $SERVER
log "RECHECK done"
