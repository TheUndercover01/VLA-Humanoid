#!/bin/bash
# 12 Oct (user: a sabotage game, player against the robot's reliability, kept simple): where does the student break? Task c1 (pick up the cube,
# put it on the target), one sabotage per episode, ours_cyl_marks_realref, 100 val layouts, lenient. Kinds and timings:
#   move@T   the cube is moved to a free spot T s into the episode (before / during the grasp)
#   drop     the cube is taken out of the hand while it is carried ("pick up" already marked done); _um1 = the game erases that mark (referee)
#   knock    the cube is thrown off the target after the task is done; _um2 = the game erases both marks
#   setsid nohup scripts/sabotage_sweep.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:-ours_cyl_marks_realref}; GPU=1; PORT=6197; CH=c1
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') SABOTAGE $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_sabotage.log 2>&1) &
S=$!
until grep -q "serving" $C/server_sabotage.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
run() {   # run TAG ARGS...
  local T=sab_${NAME}_$1; shift
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $CH --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt "$@" --name $T --out runs/eval/$T.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
  log "$NAME $T: $(grep -aoE 'lenient [0-9]+/[0-9]+' $C/ev_$T.log | tail -1) | $(grep -aoE 'disturbed [0-9]+/[0-9]+, end result reached again [0-9]+/[0-9]+, median time [0-9.na]+ s, marks at the disturbance \[[0-9, ]*\]' $C/ev_$T.log | tail -1)"
}
run none
for t in 10 20 30 40 60; do run move$t --perturb move --perturb_after $t; done
run drop --perturb drop
run drop_um1 --perturb drop --unmark 1
run knock --perturb knock
run knock_um2 --perturb knock --unmark 2
kill $S
log "$NAME done"
