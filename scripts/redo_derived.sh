#!/bin/bash
# 9 Oct (user): the headline pipeline (ours_cyl_marks) redone with ONE change to the teacher, the clip-derived reward (--derived: per-skill
# at-rest speed and hold time from keypoint_ref_real_straight_v2.npz). Teacher keypoints_cyl_derived_s1 = train_teacher.sh + --derived.
# Cut to fit the submission deadline (user, 20:40): 1200 paired + 600 push episodes (realref sizes, same 2:1 mix as 3000 + 1500), recorded on
# both GPUs; 24 build shards; the marks dataset recipe of ours_cyl_marks (MAXC=20 KN=20 DUP=0 TOK=192); SmolVLA from smolvla_base for a
# fixed STEPS=14000 (ours_cyl_marks: 34K, early-stopped; the LR schedule auto-scales to the shorter run); eval of 6 chains x 50 validation
# layouts on both GPUs (same flags as eval_memory_student.sh). Recording uses the real-clip template and the derived env.
#   setsid nohup scripts/redo_derived.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
RUN=keypoints_cyl_derived_s1; NAME=${NAME:-ours_cyl_marks_derived}; STEPS=${STEPS:-14000}; SHARDS=24; NLAY=${NLAY:-50}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python; IPY=/media/storage/ayush/miniconda3/envs/isaaclab/bin/python
REF=data/processed/keypoint_ref_real_straight_v2.npz
RA=data/processed/rollouts/${NAME}_main_a; RB=data/processed/rollouts/${NAME}_main_b; RP=data/processed/rollouts/${NAME}_push
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
trainer() { pgrep -f "isaaclab/bin/python sim/train_rl.py.*--run_name $RUN" > /dev/null; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
rec() {   # rec GPU LOG OUT EPISODES SEED: a recorder needs ~8 GB; a hung recorder (seen after an out-of-memory) is killed
  until [ $(free $1) -ge 10000 ]; do sleep 20; done
  CUDA_VISIBLE_DEVICES=$1 timeout -k 60 3600 scripts/run_isaac.sh $2 sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T0 --hold_fix --derived --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes $4 --seed $5 --out $3
  log "DERIVED $NAME recorded $(basename $3): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $2 | tail -1 | cut -c1-120)"
}

log "DERIVED $NAME: waiting for the teacher $RUN (400 iterations)"
while trainer; do sleep 30; done
CK=$(ls runs/rl/$RUN/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
[ "$CK" -ge 380 ] || { log "DERIVED $NAME: teacher ended early at model_$CK, stopping"; exit 1; }
cp runs/rl/$RUN/model_$CK.pt runs/rl/$RUN/final_teacher.pt
T0=$C/teacher_${NAME}_gpu0.pt
$IPY -c "
import torch; torch.save(torch.load('runs/rl/$RUN/final_teacher.pt', map_location='cuda:0', weights_only=False), '$T0')"
log "DERIVED $NAME: teacher model_$CK; recording 1200 paired (800 GPU 1, 400 GPU 0) + 600 push (GPU 0)"
rm -rf $RA $RB $RP
rec 1 $C/rec_${NAME}_main_a.log $RA 800 100 &
P1=$!
(VLA_PUSH_SHARE=1 rec 0 $C/rec_${NAME}_push.log $RP 600 777; rec 0 $C/rec_${NAME}_main_b.log $RB 400 101) &
P2=$!
wait $P1 $P2
for f in main_a main_b push; do grep -aq "successful episodes" $C/rec_${NAME}_$f.log || { log "DERIVED $NAME: recording $f failed"; exit 1; }; done

# informational teacher eval in the derived env while the dataset builds (GPUs idle then), not waited for
for lane in "0 c1 unstack push" "1 c2 restack lift"; do
  set -- $lane; G=$1; shift
  (for ch in "$@"; do
     CUDA_VISIBLE_DEVICES=$G timeout -k 30 1200 scripts/run_isaac.sh $C/teacher_${RUN}_$ch.log sim/eval_chains.py --headless --device cuda:0 \
       --keypoints --hold_fix --derived --ref $REF --action_filter 0.95 --chain $ch --val --num_envs $NLAY --policy rsl:$T0 \
       --name teacher_derived --out runs/eval/teacher_${RUN}_$ch.csv
     log "DERIVED teacher $RUN model_$CK [$ch, $NLAY layouts]: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+' $C/teacher_${RUN}_$ch.log | tail -1)"
   done) > /dev/null 2>&1 &
done

log "DERIVED $NAME: building $SHARDS shards (marks, pad even up to 20, k/20, dup_push 0, done channel)"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $RA $RB $RP --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode marks --pad even --dup_push 0 --max_cmds 20 --k_norm 20 --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
done
wait $(jobs -p | tail -$SHARDS)
[ "$(grep -al '^wrote' $C/build_${NAME}_s*.log | wc -l)" = "$SHARDS" ] || { log "DERIVED $NAME: shard build failed"; exit 1; }
rm -rf $D/lerobot/$NAME
HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -c "
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
D = Path('$D/lerobot')
aggregate_datasets([f'local/${NAME}_s{i}' for i in range($SHARDS)], 'local/$NAME',
                   roots=[D / 'shards' / f'${NAME}_s{i}' for i in range($SHARDS)], aggr_root=D / '$NAME')
" > $C/build_$NAME.log 2>&1
[ -f $D/lerobot/$NAME/meta/info.json ] || { log "DERIVED $NAME: aggregate failed"; exit 1; }
log "DERIVED $NAME dataset aggregated"

FT=$(( $(free 0) >= $(free 1) ? 0 : 1 ))
until [ $(free $FT) -ge 12000 ]; do sleep 20; done
log "DERIVED $NAME: training from smolvla_base, $STEPS steps (GPU $FT, token limit 192)"
rm -rf $D/train/$NAME
TOK=192 scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME $STEPS $FT > $C/ft_$NAME.log 2>&1
log "DERIVED $NAME trained: $(grep -aE '^EXIT' $C/ft_$NAME.log) $(grep -aoE 'step:[0-9.K]+ .*loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "DERIVED $NAME: no checkpoint"; exit 1; }

# eval: 2 lanes (one server + one sim per GPU), same flags as eval_memory_student.sh FLAGS=--marks, first $NLAY validation layouts
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
