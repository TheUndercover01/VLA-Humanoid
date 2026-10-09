#!/bin/bash
# 7 Oct (user: "give the WHOLE prompt at once and use the done with the whole prompt, not a single prompt"): the cube + cylinder fixed-target
# rollouts that already exist (3000 paired episodes, teacher keypoints_cyl_fixedtgt_c3half_filt095_s1) -> a dataset where every frame has the
# whole instruction sentence (padded with random commands before and after, so that sentences have up to 8 commands and the policy learns
# to do command number k + 1) and a state number k / 6 = how many commands are done (build_dataset --prompt_mode memory --done_frames 5)
# -> SmolVLA from smolvla_base with the early-stop watchdog -> eval_memory_student.sh (whole sentence every step, the count raised by
# the policy's own done flag).
#   setsid nohup scripts/distill_memory.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:-ours_cyl_memory}; ROLL=${ROLL:-data/processed/rollouts/ours_cyl_fixed_done}; SHARDS=${SHARDS:-12}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
log "$NAME: building $SHARDS shards (whole sentence + padding, state k/6, done channel) from $ROLL"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $ROLL --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode memory --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
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
log "$NAME dataset aggregated"
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
log "$NAME: fine-tuning from smolvla_base, up to 100000 steps (GPU $GPU, $bestfree MB free)"
scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 100000 $GPU > $C/ft_$NAME.log 2>&1 &
sleep 120
PID=$(pgrep -f "lerobot-train.*dataset.repo_id=local/$NAME" | head -1)
[ -n "$PID" ] && setsid nohup python3 scripts/early_stop.py --log $C/ft_$NAME.log --pid $PID --window 6000 --min_gain 0.015 >> $C/early_stop_$NAME.log 2>&1 < /dev/null &
log "$NAME: early-stop watchdog on (window 6000 steps, 1.5% gain; trainer pid ${PID:-none})"
NAME=$NAME PORT=6111 GPU=$GPU scripts/eval_memory_student.sh
