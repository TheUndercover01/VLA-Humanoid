#!/bin/bash
# 15 Oct (user: tic-tac-toe played ON the table, flat pieces): the board student ours_cyl_marks_ttt fumbles once flat X / O pieces lie on the
# board (3 of 7 robot moves placed in 3 test games; 5 of 5 on an empty board). Fine-tune it on recordings that show them.
#   1. record the board teacher keypoints_cyl_board_s1 (unchanged) with VLA_TTT_TILES=1: 0-7 random flat pieces on the squares other than the
#      target, the grid in 85% (sim/envs/ttt_tiles.py; visual only, the teacher sees privileged state): 2000 paired episodes, GPU 1
#   2. dataset = these + its own 2000 board episodes (no pieces), marks, pad 10
#   3. fine-tune FROM ours_cyl_marks_ttt, at most 12000 steps, early stop -> ours_cyl_marks_ttt_tiles
#   4. evals (100 layouts, loose check): c1 on board targets with pieces, old vs new; new without pieces; c2 with pieces
#   5. restart the live tic-tac-toe game with the new student
#   setsid nohup scripts/run_ttt_tiles_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET=board
NAME=ours_cyl_marks_ttt_tiles; G=1
T=runs/rl/keypoints_cyl_board_s1/final_teacher.pt; REF=data/processed/keypoint_ref_real_straight.npz
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
INITCK=$D/train/ours_cyl_marks_ttt/checkpoints/last/pretrained_model
LOOSE="--xy_tol 0.05 --stack_ztol 0.03 --target_z 0.06 --lift_rise 0.04 --lenient_hold 2 --ordered_hold 2"
log() { echo "$(date '+%a %H:%M') TTTTILES $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }

# ---------- 1. record ----------
RA=data/processed/rollouts/${NAME}_a; OLD=data/processed/rollouts/ours_cyl_marks_ttt_board
until [ $(free $G) -ge 10000 ]; do sleep 30; done
log "$NAME: recording 2000 paired board episodes with random flat pieces (teacher keypoints_cyl_board_s1, GPU $G)"
rm -rf $RA
VLA_TTT_TILES=1 VLA_TTT_TILES_SEED=401 CUDA_VISIBLE_DEVICES=$G timeout -k 60 7200 scripts/run_isaac.sh $C/rec_${NAME}_a.log sim/record_keypoint_rollouts.py \
  --headless --device cuda:0 --policy rsl:$T --hand_leg --rand_init 1.0 --bank_p 0 --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 \
  --paired --episodes 2000 --seed 400 --out $RA
log "$NAME recorded: $(grep -aE 'successful episodes|Traceback|OutOfMemory' $C/rec_${NAME}_a.log | tail -1 | cut -c1-120)"
grep -aq "successful episodes" $C/rec_${NAME}_a.log || { log "$NAME: recording failed"; exit 1; }

# ---------- 2. dataset ----------
SHARDS=16
log "$NAME: building $SHARDS shards (marks, pad even up to 10) from $RA $OLD"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $RA $OLD --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode marks --pad even --dup_push 0 --max_cmds 10 --k_norm 10 --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
done
wait
[ "$(grep -al '^wrote' $C/build_${NAME}_s*.log | wc -l)" = "$SHARDS" ] || { log "$NAME: shard build failed"; exit 1; }
rm -rf $D/lerobot/$NAME
HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -c "
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
D = Path('$D/lerobot')
aggregate_datasets([f'local/${NAME}_s{i}' for i in range($SHARDS)], 'local/$NAME',
                   roots=[D / 'shards' / f'${NAME}_s{i}' for i in range($SHARDS)], aggr_root=D / '$NAME')
" > $C/build_$NAME.log 2>&1
[ -f $D/lerobot/$NAME/meta/info.json ] || { log "$NAME: aggregate failed"; exit 1; }

# ---------- 3. fine-tune ----------
until [ $(free $G) -ge 12000 ]; do sleep 60; done
rm -rf $D/train/$NAME
log "$NAME: fine-tuning FROM ours_cyl_marks_ttt, at most 12000 steps (GPU $G)"
INIT=$INITCK TOK=128 scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 12000 $G > $C/ft_$NAME.log 2>&1 &
sleep 120
PID=$(pgrep -f "lerobot-train.*dataset.repo_id=local/$NAME" | head -1)
[ -n "$PID" ] && setsid nohup python3 scripts/early_stop.py --log $C/ft_$NAME.log --pid $PID --window 4000 --min_gain 0.015 >> $C/early_stop_$NAME.log 2>&1 < /dev/null &
wait
until grep -aq "^EXIT" $C/ft_$NAME.log; do sleep 30; done
log "$NAME trained: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "$NAME: no checkpoint"; exit 1; }

# ---------- 4. evals ----------
ev() {   # ev STUDENT PORT TILES CHAIN
  local tag=$(basename $(dirname $(dirname $(dirname $1))))_tiles$3_$4
  VLA_TTT_TILES=$3 VLA_TTT_TILES_SEED=7 CUDA_VISIBLE_DEVICES=$G timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$tag.log sim/eval_chains.py --headless \
    --device cuda:0 --keypoints --hold_fix --chain $4 --val --policy vla --vla_port $2 --replan 1 --learned_done --marks --whole_prompt $LOOSE \
    --name $tag --out runs/eval/$tag.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$tag.csv" 2>/dev/null
  log "$tag: $(score $C/ev_$tag.log)"
}
for ck in $D/train/$NAME/checkpoints/last/pretrained_model $INITCK; do
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD; exec $PY -m vla.server --policy $ck --port 6205 > $C/server_tttiles.log 2>&1) &
  S=$!
  until grep -q "serving" $C/server_tttiles.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
  ev $ck 6205 1 c1
  [ $ck = $INITCK ] || { ev $ck 6205 0 c1; ev $ck 6205 1 c2; }
  kill $S; sleep 5; rm -f $C/server_tttiles.log
done

# ---------- 5. live game ----------
GPU=$G play/run_tictactoe.sh start $NAME > $C/ttt_restart.log 2>&1
log "$NAME: pipeline done; live tic-tac-toe restarted with it ($(grep -a 'tictactoe on' $C/ttt_restart.log))"
