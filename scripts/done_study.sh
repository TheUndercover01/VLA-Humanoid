#!/bin/bash
# 8 Oct, "the policy's done flag runs ahead of what happened" (the pointer is at 12.6/20 after about 1.5 commands in order): can the long-chain
# failure be fixed without training? (1) AUDIT: at every pointer advance, was that command's outcome really true (eval_chains --done_diag);
# (2) SWEEP the flag's threshold / hold time on the marks student; (3) VOTE: the flag = the minimum over 3 samples of the VLA.
#   MODE=smoke|full setsid nohup scripts/done_study.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
MODE=${MODE:-full}; GPU=${GPU:-1}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
serve() {   # name port -> starts the server, sets SERVER
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
   exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$1/checkpoints/last/pretrained_model --port $2 > $C/server_ds_$1.log 2>&1) &
  SERVER=$!
  until grep -q "serving" $C/server_ds_$1.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
  kill -0 $SERVER 2>/dev/null || { log "DONESTUDY $1: server failed"; exit 1; }
}
run() {     # name port flags tag chain thresh hold vote envs
  local name=$1 port=$2 flags=$3 tag=$4 ch=$5 th=$6 hold=$7 vote=$8 envs=$9
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 2400 scripts/run_isaac.sh $C/ds_${name}_${tag}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --num_envs $envs --policy vla --vla_port $port --replan 1 --learned_done $flags --whole_prompt \
    --done_thresh $th --done_hold $hold --done_vote $vote --done_diag --name ${name}_ds --out runs/eval/ds_${name}_${tag}_$ch.csv
  flock -u 9; exec 9>&-
  pkill -9 -f "sim/eval_chains.py.*ds_${name}_${tag}_$ch" 2>/dev/null   # an Isaac process can hang at shutdown
  log "DONESTUDY $name $tag $ch (thresh $th, hold $hold, vote $vote): $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ds_${name}_${tag}_$ch.log | tail -1) | $(grep -aoE 'done flag audit: .*' $C/ds_${name}_${tag}_$ch.log | tail -1)"
}
if [ $MODE = smoke ]; then
  serve ours_cyl_marks 6151; run ours_cyl_marks 6151 --marks smoke rand4_1 0.85 5 2 4; kill $SERVER; exit 0
fi
CH_SHORT="rand4_1 rand4_0 rand8_1 unstack"
# 1. audit at the default flag, both students
serve ours_cyl_marks 6151
for ch in $CH_SHORT restack; do run ours_cyl_marks 6151 --marks base $ch 0.85 5 1 $([ $ch = rand4_1 -o $ch = rand4_0 -o $ch = rand8_1 ] && echo 50 || echo 100); done
# 2. sweep of the flag on the marks student
for cfg in "0.85 8" "0.9 8" "0.95 8" "0.95 12"; do
  set -- $cfg
  for ch in $CH_SHORT; do run ours_cyl_marks 6151 --marks t$1h$2 $ch $1 $2 1 $([ $ch = unstack ] && echo 100 || echo 50); done
done
# 3. vote of 3 samples at the default flag
for ch in $CH_SHORT; do run ours_cyl_marks 6151 --marks vote3 $ch 0.85 5 3 $([ $ch = unstack ] && echo 100 || echo 50); done
kill $SERVER
# the number student: audit only
serve ours_cyl_mem_even 6152
for ch in rand4_1 rand8_1 unstack; do run ours_cyl_mem_even 6152 "--memory_done --k_norm 20" base $ch 0.85 5 1 $([ $ch = unstack ] && echo 100 || echo 50); done
kill $SERVER
log "DONESTUDY done"
