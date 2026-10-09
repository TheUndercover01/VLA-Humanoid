#!/bin/bash
# 9 Oct 21:40: second half of redo_derived.sh, started while the recorders were still running (the first driver waited for ALL recording
# before building anything). Each recorded set is built into its own shards as soon as it is finished (main_a done at 21:24, push, main_b),
# the shards are merged at the end; then training, teacher eval and student eval are exactly as in redo_derived.sh.
# The recorders (and the push -> main_b chain) are not touched.
#   setsid nohup scripts/redo_derived2.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
RUN=keypoints_cyl_derived_s1; NAME=${NAME:-ours_cyl_marks_derived}; STEPS=${STEPS:-14000}; NLAY=${NLAY:-50}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
REF=data/processed/keypoint_ref_real_straight_v2.npz
T0=$C/teacher_${NAME}_gpu0.pt; CK=399
RA=data/processed/rollouts/${NAME}_main_a; RB=data/processed/rollouts/${NAME}_main_b; RP=data/processed/rollouts/${NAME}_push
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
recorded() { grep -aq "successful episodes" $C/rec_${NAME}_$1.log 2>/dev/null; }

PARTS=""
build_part() {   # build_part TAG SRC SHARDS: builds SHARDS shards of one recorded set in the background
  local tag=$1 src=$2 n=$3
  log "DERIVED $NAME: building part $tag ($n shards) from $src"
  for i in $(seq 0 $((n-1))); do
    HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $src --repo_id local/${NAME}_${tag}_s$i \
      --root $D/lerobot/shards/${NAME}_${tag}_s$i --prompt_mode marks --pad even --dup_push 0 --max_cmds 20 --k_norm 20 --done_frames 5 --streaming --shard $i/$n > $C/build_${NAME}_${tag}_s$i.log 2>&1 &
  done
  PARTS="$PARTS $tag:$n"
}
rm -rf $D/lerobot/shards/${NAME}_*_s* $D/lerobot/shards/${NAME}_s*
build_part a $RA 12
until recorded push; do sleep 10; done
log "DERIVED $NAME push recorded: $(grep -aE 'successful episodes' $C/rec_${NAME}_push.log | tail -1 | cut -c1-100)"
build_part p $RP 8
until recorded main_b; do sleep 10; done
log "DERIVED $NAME main_b recorded: $(grep -aE 'successful episodes' $C/rec_${NAME}_main_b.log | tail -1 | cut -c1-100)"
build_part b $RB 6

# informational teacher eval in the derived env (not waited for)
for lane in "0 c1 unstack push" "1 c2 restack lift"; do
  set -- $lane; G=$1; shift
  (for ch in "$@"; do
     CUDA_VISIBLE_DEVICES=$G timeout -k 30 1200 scripts/run_isaac.sh $C/teacher_${RUN}_$ch.log sim/eval_chains.py --headless --device cuda:0 \
       --keypoints --hold_fix --derived --ref $REF --action_filter 0.95 --chain $ch --val --num_envs $NLAY --policy rsl:$T0 \
       --name teacher_derived --out runs/eval/teacher_${RUN}_$ch.csv
     log "DERIVED teacher $RUN model_$CK [$ch, $NLAY layouts]: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+' $C/teacher_${RUN}_$ch.log | tail -1)"
   done) > /dev/null 2>&1 &
done

# wait only for the shard builders (the teacher-eval lanes keep running)
while pgrep -f "vla.build_dataset.*${NAME}_" > /dev/null; do sleep 5; done
ROOTS=""; REPOS=""; TOTAL=0
for pr in $PARTS; do
  tag=${pr%%:*}; n=${pr##*:}
  [ "$(grep -al '^wrote' $C/build_${NAME}_${tag}_s*.log | wc -l)" = "$n" ] || { log "DERIVED $NAME: shard build failed (part $tag)"; exit 1; }
  for i in $(seq 0 $((n-1))); do REPOS="$REPOS 'local/${NAME}_${tag}_s$i',"; ROOTS="$ROOTS D / 'shards' / '${NAME}_${tag}_s$i',"; TOTAL=$((TOTAL+1)); done
done
rm -rf $D/lerobot/$NAME
HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -c "
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
D = Path('$D/lerobot')
aggregate_datasets([$REPOS], 'local/$NAME', roots=[$ROOTS], aggr_root=D / '$NAME')
" > $C/build_$NAME.log 2>&1
[ -f $D/lerobot/$NAME/meta/info.json ] || { log "DERIVED $NAME: aggregate failed"; exit 1; }
log "DERIVED $NAME dataset aggregated ($TOTAL shards)"

FT=$(( $(free 0) >= $(free 1) ? 0 : 1 ))
until [ $(free $FT) -ge 12000 ]; do sleep 20; done
log "DERIVED $NAME: training from smolvla_base, $STEPS steps (GPU $FT, token limit 192)"
rm -rf $D/train/$NAME
TOK=192 scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME $STEPS $FT > $C/ft_$NAME.log 2>&1
log "DERIVED $NAME trained: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.K]+ .*loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "DERIVED $NAME: no checkpoint"; exit 1; }

lane() {   # lane GPU PORT chains...
  local G=$1 P=$2; shift 2
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
   exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $P > $C/server_${NAME}_$G.log 2>&1) &
  local S=$!
  until grep -q "serving" $C/server_${NAME}_$G.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
  for ch in "$@"; do
    CUDA_VISIBLE_DEVICES=$G timeout -k 30 1500 scripts/run_isaac.sh $C/mem_${NAME}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
      --hold_fix --chain $ch --val --num_envs $NLAY --policy vla --vla_port $P --replan 1 --learned_done --marks --whole_prompt \
      --name ${NAME}_mem --out runs/eval/${NAME}_last_mem_${ch}_val.csv
    log "VLA $NAME [WHOLE prompt + marks, $NLAY layouts] $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/mem_${NAME}_$ch.log | tail -1)"
  done
  kill $S
}
lane 0 6181 unstack c1 push &
lane 1 6182 restack c2 lift &
wait
log "DERIVED $NAME: done"
