#!/bin/bash
# 13 Oct (user: tic-tac-toe, the target on "some specific places", not random): one per-game fine-tune of the best short-task student.
# GPU 1 only. Waits until the long-horizon diagnosis (LONGDIAG) and the no-lift pipeline's recording no longer need GPU 1.
#   1. teacher keypoints_cyl_board_s1: the no-lift recipe (hand leg, push-lift penalty 0.3, no scripted starts, random start pose) with the
#      target on one of the 9 tic-tac-toe square centres (VLA_FIXED_TARGET=board, sim/envs/tasks.py), 1000 iterations x 16384 envs.
#   2. gate: c1 / c2 / push / lift on board targets, 100 layouts; pass = every training skill >= 0.9 and c1 >= 90.
#   3. record 2000 paired + 500 push episodes on board targets (bank_p 0, random start), fine-tune ours_cyl_marks_handleg_rand on them plus its
#      own 1500 main_a episodes (old target spot, against forgetting) -> ours_cyl_marks_ttt (marks, pad 10, TOK 128, early stop, at most 30K steps).
#   4. evals: board targets c1 / c1_cyl / c2 / push (the game), old spot c1 / c2 / unstack / restack (forgetting), 100 layouts each.
#   setsid nohup scripts/run_ttt_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET=board
RUN=keypoints_cyl_board_s1; NAME=ours_cyl_marks_ttt; G=1; RI=1.0; ITERS=${ITERS:-1000}
REF=data/processed/keypoint_ref_real_straight.npz
INITCK=/media/storage/ayush/vla_data/train/ours_cyl_marks_handleg_rand/checkpoints/last/pretrained_model
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') TTT $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }

# ---------- 0. wait for GPU 1 ----------
until [ $(grep -ac "LONGDIAG .* done (tables" $STATUS) -ge 2 ]; do sleep 60; done
until grep -aq "NOLIFT ours_cyl_marks_nolift recorded ours_cyl_marks_nolift_main_a\|NOLIFT gate failed\|NOLIFT teacher .* ended early" $STATUS; do sleep 60; done
until [ $(free $G) -ge 20000 ]; do sleep 60; done

# ---------- 1. teacher ----------
if [ ! -f runs/rl/$RUN/model_$((ITERS-1)).pt ]; then
  RES=$(ls runs/rl/$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)
  log "training $RUN${RES:+ (resumed from $(basename $RES))}: no-lift recipe, target on the 9 board squares, $ITERS iterations x 16384 envs (GPU $G)"
  CUDA_VISIBLE_DEVICES=$G scripts/run_isaac.sh $C/train_$RUN.log sim/train_rl.py --headless --device cuda:0 \
    --task keypoints --seed 1 --num_envs 16384 --max_iterations $ITERS --run_name $RUN --max_chain 2 --hold_fix \
    --action_filter 0.95 --episode_s 20 --save_interval 20 --ref $REF --bank data/processed/state_bank_standin.npz \
    --hand_leg --push_lift_pen 0.3 --bank_p 0 --rand_init $RI ${RES:+--resume $RES}
fi
CK=$(ls runs/rl/$RUN/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
[ "$CK" -ge $((ITERS-20)) ] || { log "teacher ended early at model_$CK, stopping (see $C/train_$RUN.log)"; exit 1; }
cp runs/rl/$RUN/model_$CK.pt runs/rl/$RUN/final_teacher.pt
T=runs/rl/$RUN/final_teacher.pt
SK=$(python3 - $C/train_$RUN.log <<'EOF2'
import re, sys
t = open(sys.argv[1], errors="ignore").read()
print(" ".join(f"{k}={sum(v) / max(len(v), 1):.3f}" for k in ("push", "pick-lift", "place-down")
               for v in [[float(x) for x in re.findall(rf"Skill/{k}: ([0-9.]+)", t)][-10:]]))
EOF2
)
log "teacher $RUN model_$CK done; training skills (mean of the last 10 iterations): $SK"

# ---------- 2. gate ----------
for ch in c1 c2 push lift; do
  CUDA_VISIBLE_DEVICES=$G timeout -k 30 3600 scripts/run_isaac.sh $C/tev_board_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs 100 --policy rsl:$T --hand_leg \
    --name board --out runs/eval/teacher_board_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/teacher_board_$ch.csv" 2>/dev/null
  log "teacher board [$ch, 100 layouts, board targets]: $(score $C/tev_board_$ch.log)"
done
C1=$(grep -aoE 'lenient [0-9]+/100' $C/tev_board_c1.log | tail -1 | grep -oE '[0-9]+' | head -1)
OK=$(python3 -c "s='$SK'; print(all(float(v.split('=')[1]) >= 0.9 for v in s.split()) and int('${C1:-0}') >= 90)")
[ "$OK" = True ] || { log "gate failed (skills $SK, c1 ${C1:-?}/100), no student"; exit 1; }
log "gate passed"

# ---------- 3. record + fine-tune ----------
RA=data/processed/rollouts/${NAME}_board; RP=data/processed/rollouts/${NAME}_board_push
OLD=data/processed/rollouts/ours_cyl_marks_handleg_rand_main_a
rec() {   # rec OUT EPISODES SEED
  until [ $(free $G) -ge 10000 ]; do sleep 20; done
  local L=$C/rec_$(basename $1).log
  CUDA_VISIBLE_DEVICES=$G timeout -k 60 7200 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T --hand_leg --rand_init $RI --bank_p 0 --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes $2 --seed $3 --out $1
  log "$NAME recorded $(basename $1): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $L | tail -1 | cut -c1-120)"
  grep -aq "successful episodes" $L || { log "$NAME: recording $(basename $1) failed"; exit 1; }
}
rm -rf $RA $RP
rec $RA 2000 300
VLA_PUSH_SHARE=1 rec $RP 500 301
SHARDS=16
log "$NAME: building $SHARDS shards (marks, pad even up to 10) from $RA $RP $OLD"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $RA $RP $OLD --repo_id local/${NAME}_s$i \
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
until [ $(free $G) -ge 12000 ]; do sleep 60; done
rm -rf $D/train/$NAME
log "$NAME: fine-tuning FROM ours_cyl_marks_handleg_rand, at most 30000 steps (GPU $G, token limit 128)"
INIT=$INITCK TOK=128 scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 30000 $G > $C/ft_$NAME.log 2>&1 &
sleep 120
PID=$(pgrep -f "lerobot-train.*dataset.repo_id=local/$NAME" | head -1)
[ -n "$PID" ] && setsid nohup python3 scripts/early_stop.py --log $C/ft_$NAME.log --pid $PID --window 6000 --min_gain 0.015 >> $C/early_stop_$NAME.log 2>&1 < /dev/null &
wait
until grep -aq "^EXIT" $C/ft_$NAME.log; do sleep 30; done
log "$NAME trained: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "$NAME: no checkpoint"; exit 1; }

# ---------- 4. evals (GPU 1) ----------
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port 6199 > $C/server_$NAME.log 2>&1) &
S=$!
until grep -q "serving" $C/server_$NAME.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
ev() {   # ev TARGETS CHAIN
  local T_=${NAME}_$1_$2
  VLA_FIXED_TARGET=$([ $1 = board ] && echo board || echo "0.03,-0.12") CUDA_VISIBLE_DEVICES=$G timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T_.log \
    sim/eval_chains.py --headless --device cuda:0 --keypoints --hold_fix --chain $2 --val --policy vla --vla_port 6199 --replan 1 --learned_done \
    --marks --whole_prompt --name $T_ --out runs/eval/$T_.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$T_.csv" 2>/dev/null
  log "VLA $NAME [$1 targets] $2: $(score $C/ev_$T_.log)"
}
for ch in c1 c1_cyl c2 push; do ev board $ch; done
for ch in c1 c2 unstack restack; do ev oldspot $ch; done
kill $S
log "$NAME: pipeline done"
