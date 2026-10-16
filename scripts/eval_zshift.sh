#!/bin/bash
# 12 Oct (user: "a z test for the same stacking method"; the viewer sees the real scene, only what the robot is shown changes): c2 (pick up the
# cube, put the cube on the cylinder) with the cylinder standing on a fixed column of height H (a tower of cubes of 5 cm each under the base).
# The VLA is shown the scene lowered by H while it carries the cube to the put-down: the front camera moves up by H and the TCP height told to it
# drops by H (sim/vla_policy.py place_shift, switched at the (done) mark), so the cylinder looks like it stands on the table, where the policy was
# trained. During the pick-up nothing is shifted. ours_cyl_marks_realref, 100 val layouts, lenient.
#   setsid nohup scripts/eval_zshift.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12" VLA_PEDESTAL=1
NAME=ours_cyl_marks_realref; GPU=1; PORT=6196; CH=${CH:-c2}
LEVELS=${LEVELS:-"0 0.05 0.10 0.15 0.20 0.25"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_zshift.log 2>&1) &
S=$!
until grep -q "serving" $C/server_zshift.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for H in $LEVELS; do
  for mode in shown raw; do        # shown: the VLA gets the lowered scene; raw: no shift at all (what it does without the layer)
    [ $mode = raw ] && [ "$H" = 0 ] && continue
    T=zshift_${CH}_${H}_$mode
    PS=$([ $mode = shown ] && echo "--place_shift=0,0,$H" || echo "")
    CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
      --hold_fix --chain $CH --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --pedestal $H $PS \
      --name $T --out runs/eval/$T.csv
    pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
    log "ZSHIFT $NAME $CH column $H m, $mode: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_$T.log | tail -1)"
  done
done
kill $S
log "ZSHIFT $NAME done"
