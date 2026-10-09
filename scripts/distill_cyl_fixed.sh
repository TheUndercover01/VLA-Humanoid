#!/bin/bash
# 6 Oct night (user asleep): the fixed-target cube + cylinder teacher trains for 400 iterations (train_rl.py --max_iterations 400),
# then, with no score gate ("the done thing in the training data is the teacher's own command switching, not the lenient score"):
# 3000 paired episodes recorded from its last checkpoint -> done-channel dataset (distill_done.sh recipe: one-command prompts,
# --done_frames 5) -> SmolVLA from smolvla_base, up to 100k steps with the early-stop watchdog -> eval_done_student.sh (VLM's own done flag moves the chain).
# An informational all-chain eval of the teacher checkpoint runs in the background on GPU 0 (not a gate).
#   setsid nohup scripts/distill_cyl_fixed.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
RUN=${RUN:-keypoints_cyl_fixedtgt_c3half_filt095_s1}; NAME=${NAME:-ours_cyl_fixed_done}; EPISODES=${EPISODES:-3000}; SHARDS=${SHARDS:-12}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python; IPY=/media/storage/ayush/miniconda3/envs/isaaclab/bin/python
ROLL=data/processed/rollouts/$NAME
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
trainer() { pgrep -f "isaaclab/bin/python sim/train_rl.py.*--run_name $RUN" > /dev/null; }

log "$NAME: waiting for the teacher $RUN (400 iterations)"
for i in $(seq 1 20); do trainer && break; sleep 30; done
trainer || { log "$NAME: the teacher run is not running"; exit 1; }
while trainer; do sleep 60; done
CK=$(ls runs/rl/$RUN/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
log "$NAME: teacher run ended, last checkpoint model_$CK.pt; recording $EPISODES paired episodes"
cp runs/rl/$RUN/model_$CK.pt runs/rl/$RUN/final_teacher.pt

# informational all-chain eval of the teacher (not a gate), GPU 0
(for ch in push push_cyl lift lift_cyl c1 c1_cyl c2 c3 swap unstack restack; do
   CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/final_teacher_${RUN}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints --hold_fix \
     --action_filter 0.95 --chain $ch --val --policy rsl:runs/rl/$RUN/final_teacher.pt --name final_teacher --out $C/final_teacher_${RUN}_$ch.csv
   log "teacher $RUN model_$CK [$ch]: $(grep -aoE 'lenient [0-9]+/100, in-order [0-9]+/100' $C/final_teacher_${RUN}_$ch.log | tail -1)"
 done) > /dev/null 2>&1 &

T0=$C/teacher_${NAME}_gpu0.pt
$IPY -c "
import torch; torch.save(torch.load('runs/rl/$RUN/final_teacher.pt', map_location='cuda:0', weights_only=False), '$T0')"
exec 9>$C/isaac_eval.lock; flock 9
CUDA_VISIBLE_DEVICES=${REC_GPU:-1} scripts/run_isaac.sh $C/rec_$NAME.log sim/record_keypoint_rollouts.py --headless --device cuda:0 \
  --policy rsl:$T0 --hold_fix --action_filter 0.95 --episode_s 20 --paired --episodes $EPISODES --out $ROLL
flock -u 9; exec 9>&-
log "$NAME recorded: $(grep -aE 'successful episodes|Traceback' $C/rec_$NAME.log | tail -1)"
grep -aq "successful episodes" $C/rec_$NAME.log || exit 1

log "$NAME: building $SHARDS shards (step prompts, done channel on the last 5 frames per command)"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $ROLL --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode step --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
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
log "$NAME dataset aggregated: $(tail -1 $C/build_$NAME.log | cut -c1-120)"

while true; do
  best=-1; bestfree=0
  for g in 0 1; do
    free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $g) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g) ))
    [ $free -gt $bestfree ] && { best=$g; bestfree=$free; }
  done
  [ $bestfree -ge 12000 ] && break
  sleep 120
done
GPU=$best
log "$NAME: fine-tuning from smolvla_base, 100000 steps (GPU $GPU, $bestfree MB free)"
scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 100000 $GPU > $C/ft_$NAME.log 2>&1 &
sleep 120
# early stopping (user: "the loss saturates, stop after some steps, 100k would overfit"): scripts/early_stop.py ends the run when the
# mean loss of the last 6000 steps is less than 1.5% below that of the 6000 before (needs 12000 steps), when the log is silent
# for 20 min, or on NaN. Checkpoints come every 2000 steps; the eval uses the last one.
PID=$(pgrep -f "lerobot-train.*dataset.repo_id=local/$NAME" | head -1)
[ -n "$PID" ] && setsid nohup python3 scripts/early_stop.py --log $C/ft_$NAME.log --pid $PID --window 6000 --min_gain 0.015 >> $C/early_stop_$NAME.log 2>&1 < /dev/null &
log "$NAME: early-stop watchdog on (window 6000 steps, 1.5% gain; trainer pid ${PID:-none})"
CHAINS="push lift lift_cyl c1 c1_cyl c2 c3 swap unstack restack" scripts/eval_done_student.sh $NAME 6105 $GPU
