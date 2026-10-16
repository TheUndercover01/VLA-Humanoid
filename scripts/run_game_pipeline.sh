#!/bin/bash
# 10 Oct (user: ONE teacher and ONE student for every game): the single game teacher (scripts/train_game_teacher.sh, run game_mix_s2) ->
# gate on 21 scene sets (the original cube + cylinder chains, boards with 0/4/7 static pieces, towers of 1-6, free chains) -> record the
# mixture (5000 episodes + 1500 extra push episodes of the original scenes) -> marks dataset (whole sentence padded to 20 commands, the policy's
# own "(done)" marks) -> ONE SmolVLA from smolvla_base with the early-stop watchdog -> the student on the same sets.
#   setsid nohup scripts/run_game_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=multi VLA_GAME=mix VLA_TRAIN_TYPES="0,1,3,4" VLA_HOLDOUT=1 VLA_FIXED_TARGET="0.03,-0.12" VLA_PUSH_C3=0.5
RUN=${RUN:-game_mix_s2}; NAME=${NAME:-ours_game_marks}; SHARDS=24; TOK=${TOK:-256}
SETS=${SETS:-"orig_pick orig_pick_cyl orig_c1 orig_c1_cyl orig_push orig_c2 orig_swap orig_c3 orig_unstack orig_restack board0 board4 board7 tower1 tower2 tower3 tower4 tower5 tower6 chain4 chain6"}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
REF=data/processed/keypoint_ref_real_straight.npz
RA=data/processed/rollouts/${NAME}_a; RB=data/processed/rollouts/${NAME}_b; RP=data/processed/rollouts/${NAME}_push
log() { echo "$(date '+%a %H:%M') GAMEPIPE $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE '\(.* commands, [0-9]+ scenes\): lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+\)' $1 | tail -1; }

# ---------- 1. the teacher ----------
log "waiting for the teacher $RUN"
until ! ps -eo cmd | grep -q "[t]rain_rl.py.*$RUN"; do sleep 30; done
CK=$(ls runs/rl/$RUN/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
[ "$CK" -ge 980 ] || { log "teacher ended early at model_$CK, stopping"; exit 1; }
cp runs/rl/$RUN/model_$CK.pt runs/rl/$RUN/final_teacher.pt
T0=$C/teacher_${NAME}.pt; cp runs/rl/$RUN/final_teacher.pt $T0
log "teacher $RUN model_$CK; gate on $(echo $SETS | wc -w) sets (100 scenes each, both GPUs)"

# ---------- 2. gate: the teacher on every set ----------
teval() {   # teval GPU SET
  CUDA_VISIBLE_DEVICES=$1 timeout -k 30 1500 scripts/run_isaac.sh $C/gg_${NAME}_$2.log sim/eval_multi.py --headless --device cuda:0 --set $2 \
    --policy rsl:$T0 --name teacher_game --out runs/eval/game_teacher_$2.csv
  pkill -9 -f "sim/eval_multi.py.*game_teacher_$2.csv" 2>/dev/null
  log "GATE teacher $2: $(score $C/gg_${NAME}_$2.log) | collateral $(python3 -c "
import csv; r=list(csv.DictReader(open('runs/eval/game_teacher_$2.csv'))); print(sum(int(x.get('collateral',0)) for x in r), 'of', len(r))" 2>/dev/null)"
}
i=0; L0=""; L1=""
for st in $SETS; do if [ $((i % 2)) = 0 ]; then L0="$L0 $st"; else L1="$L1 $st"; fi; i=$((i+1)); done
(for st in $L0; do teval 0 $st; done) &
P0=$!
(for st in $L1; do teval 1 $st; done) &
P1=$!
wait $P0 $P1
log "gate done"

# ---------- 3. record ----------
rec() {   # rec GPU OUT EPISODES SEED
  until [ $(free $1) -ge 10000 ]; do sleep 20; done
  local L=$C/rec_$(basename $2).log
  CUDA_VISIBLE_DEVICES=$1 timeout -k 60 7200 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T0 --hand_leg --bank_p 0 --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes $3 --seed $4 --out $2
  log "recorded $(basename $2): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $L | tail -1 | cut -c1-120)"
}
log "recording: 2 x 2500 mixture episodes (GPU 1 seed 100, GPU 0 seed 101) + 1500 extra push episodes (GPU 0, seed 777)"
rm -rf $RA $RB $RP
rec 1 $RA 2500 100 &
P1=$!
(VLA_PUSH_SHARE=1 rec 0 $RP 1500 777; rec 0 $RB 2500 101) &
P2=$!
wait $P1 $P2
for r in $RA $RB $RP; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "recording $(basename $r) failed"; exit 1; }; done

# ---------- 4. dataset ----------
log "building $SHARDS shards (marks, pad even up to 20, k/20, done channel, token limit $TOK)"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $RA $RB $RP --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode marks --pad even --dup_push 0 --max_cmds 20 --k_norm 20 --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
done
wait $(jobs -p | tail -$SHARDS)
[ "$(grep -al '^wrote' $C/build_${NAME}_s*.log | wc -l)" = "$SHARDS" ] || { log "shard build failed"; exit 1; }
rm -rf $D/lerobot/$NAME
HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -c "
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
D = Path('$D/lerobot')
aggregate_datasets([f'local/${NAME}_s{i}' for i in range($SHARDS)], 'local/$NAME',
                   roots=[D / 'shards' / f'${NAME}_s{i}' for i in range($SHARDS)], aggr_root=D / '$NAME')
" > $C/build_$NAME.log 2>&1
[ -f $D/lerobot/$NAME/meta/info.json ] || { log "aggregate failed"; exit 1; }
log "dataset aggregated"

# ---------- 5. the ONE student ----------
FT=$(( $(free 0) >= $(free 1) ? 0 : 1 ))
until [ $(free $FT) -ge 12000 ]; do sleep 20; done
export TOK
log "fine-tuning from smolvla_base, up to 100000 steps (GPU $FT, token limit $TOK), early stop on"
rm -rf $D/train/$NAME
scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 100000 $FT > $C/ft_$NAME.log 2>&1 &
sleep 120
PID=$(pgrep -f "lerobot-train.*dataset.repo_id=local/$NAME" | head -1)
[ -n "$PID" ] && setsid nohup python3 scripts/early_stop.py --log $C/ft_$NAME.log --pid $PID --window 6000 --min_gain 0.015 >> $C/early_stop_$NAME.log 2>&1 < /dev/null &
until grep -aq "^EXIT" $C/ft_$NAME.log 2>/dev/null; do sleep 60; done
log "student trained: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "no checkpoint"; exit 1; }

# ---------- 6. the student on every set (two lanes, a server each) ----------
seval() {   # seval GPU PORT SET
  CUDA_VISIBLE_DEVICES=$1 timeout -k 30 2400 scripts/run_isaac.sh $C/ge_${NAME}_$3.log sim/eval_multi.py --headless --device cuda:0 --set $3 \
    --policy vla --vla_port $2 --learned_done --marks --name $NAME --out runs/eval/game_${NAME}_$3.csv
  pkill -9 -f "sim/eval_multi.py.*game_${NAME}_$3.csv" 2>/dev/null
  log "STUDENT $NAME $3: $(score $C/ge_${NAME}_$3.log) | collateral $(python3 -c "
import csv; r=list(csv.DictReader(open('runs/eval/game_${NAME}_$3.csv'))); print(sum(int(x.get('collateral',0)) for x in r), 'of', len(r))" 2>/dev/null)"
}
lane() {   # lane GPU PORT sets...
  local G=$1 P=$2; shift 2
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
   exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $P > $C/server_ge_${NAME}_$G.log 2>&1) &
  local S=$!
  until grep -q "serving" $C/server_ge_${NAME}_$G.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
  for st in "$@"; do seval $G $P $st; done
  kill $S
}
lane 0 6211 $L0 &
lane 1 6212 $L1 &
wait
log "pipeline done: student $NAME on the same sets as the gate"
