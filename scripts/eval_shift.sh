#!/bin/bash
# 12 Oct (user: "manipulate around the original idea to make an application layer on top that we can play games on"): the game-layer test.
# The headline student (ours_cyl_marks_realref, trained with the target always at (0.03, -0.12)) is evaluated on c1 (pick the cube, put it on
# the target) with the WHOLE scene moved by (dx, dy): cubes and target in the sim are moved, the front camera moves by the same amount and the
# TCP position told to the VLA is shifted back (sim/eval_chains.py --shift, vla_policy.shift), so the policy sees the scene where it was
# trained and only the robot base, the table edge and the arm's pose look different. Actions are TCP deltas: no transform. 100 val layouts each.
#   setsid nohup scripts/eval_shift.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=ours_cyl_marks_realref; GPU=1; PORT=6195; CH=${CH:-c1}
SHIFTS=${SHIFTS:-"0,0 0.09,0 -0.09,0 0,0.09 0,-0.09 0.09,0.09 -0.09,-0.09"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_shift.log 2>&1) &
S=$!
until grep -q "serving" $C/server_shift.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for sh in $SHIFTS; do
  T=shift_${CH}_${sh/,/_}
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $CH --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --shift=$sh \
    --name $T --out runs/eval/$T.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
  log "SHIFT $NAME $CH shift $sh: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_$T.log | tail -1)"
done
kill $S
log "SHIFT $NAME done"
