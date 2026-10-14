#!/bin/bash
# 13 Oct (user: "diagnose the long horizon problem"): why does ours_cyl_marks_handleg_rand do 1-2 command tasks at 96-100 but unstack 30,
# restack 1, c3 10 in order? Plan: ~/.claude/plans/compressed-prancing-pearl.md. All inference, GPU 1, 100 val layouts, status lines LONGDIAG.
#   flag    the normal eval + --done_diag --done_trace (marks from the student's own flag)
#   oracle  the sim writes the marks (memory taken out: what is left is the motor skill)
#   premark unstack starting with commands 1-2 already marked, from home (commands 3-4 from a fresh start)
#   rearm   after a mark the flag must drop below 0.5 before the next one counts (inference-only fix candidate)
#   setsid nohup scripts/diagnose_long.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:-ours_cyl_marks_handleg_rand}; GPU=1; PORT=6198; NL=${NL:-100}; CHAINS=${CHAINS:-"unstack restack c3"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') LONGDIAG $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_longdiag.log 2>&1) &
S=$!
until grep -q "serving" $C/server_longdiag.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
OUTS=""
run() {   # run TAG CHAIN ARGS...
  local T=longdiag_${1}_$2 ch=$2; shift 2
  until [ $(free $GPU) -ge 8000 ]; do sleep 20; done
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --num_envs $NL --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --done_trace "$@" \
    --name $T --out runs/eval/$T.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
  OUTS="$OUTS runs/eval/$T.csv"
  log "$NAME $T: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_$T.log | tail -1) | $(grep -aoE 'done flag audit: .*' $C/ev_$T.log | tail -1 | cut -c1-150)"
}
for ch in $CHAINS; do run flag $ch --done_diag; done
for ch in $CHAINS; do run oracle $ch --perturb oracle; done
run premark unstack --perturb premark --unmark 2
for ch in $CHAINS; do run rearm $ch --done_diag --rearm 0.5; done
kill $S
python3 -m analysis.done_audit $OUTS > $C/longdiag_tables.txt 2>&1
log "$NAME done (tables: $C/longdiag_tables.txt)"
