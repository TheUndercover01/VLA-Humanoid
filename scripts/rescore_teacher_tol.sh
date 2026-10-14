#!/bin/bash
# 14 Oct (user: score with the teacher's own definition): the false put-down marks of the student sit 1.5-2.6 cm above the stack (c2, MARKPOS),
# the teacher's done check (= where the done labels are) accepts up to ~2.1 cm, the eval's in-order check only 1.5 cm. Re-evaluate the existing
# students with --stack_ztol 0.0214 (teacher's tolerance) on c2 / swap / unstack / restack / restack8 (+ restack8 strict, it is new). Eval only.
#   lane: GPU, students. Status lines RESCORE.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') RESCORE $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }
lane() {   # lane GPU PORT NAME FLAGS
  local G=$1 P=$2 N=$3 F=$4
  until [ $(free $G) -ge 7000 ]; do sleep 30; done
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
   exec $PY -m vla.server --policy $D/train/$N/checkpoints/last/pretrained_model --port $P > $C/server_rescore_$N.log 2>&1) &
  local S=$!
  until grep -q "serving" $C/server_rescore_$N.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
  for run in "c2 0.0214" "swap 0.0214" "unstack 0.0214" "restack 0.0214" "restack8 0.0214" "restack8 0.015"; do
    set -- $run
    local T=rescore_${N}_$1_$2
    until [ $(free $G) -ge 4000 ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES=$G timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
      --hold_fix --chain $1 --val --policy vla --vla_port $P --replan 1 --learned_done --marks $F --whole_prompt --stack_ztol $2 --name $T --out runs/eval/$T.csv
    pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
    log "$N $1 [stack tolerance $2 m]: $(score $C/ev_$T.log)"
  done
  kill $S
}
# 14 Oct 02:05: all on GPU 1 after the tic-tac-toe pipeline (GPU 0 trains the clip-tolerance teacher from scratch)
(until grep -aq "TTT ours_cyl_marks_ttt: pipeline done\|TTT .*failed\|TTT .*no checkpoint" $STATUS; do sleep 30; done
 lane 1 6206 ours_cyl_marks_handleg_rand ""; lane 1 6206 ours_cyl_marks_nolift ""; lane 1 6206 ours_cyl_marks_progress "--progress") &
wait
log "done"
