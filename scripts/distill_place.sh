#!/bin/bash
# 8 Oct, "how should a VLA keep its place in a long instruction?": same rollouts and recipe as distill_memory.sh, with the padding fixed so
# that the done count k covers 0..5 (build_dataset --pad even; before, "do the 6th command" never occurred in training) and every push
# episode added once more (--dup_push 1, push was the weakest skill). MODE=memory: k / 6 in the state. MODE=marks: no number, the
# commands already done carry "(done)" in the sentence. In both, the policy's own done flag is the only thing that moves its place.
# The user then asked for more push data first: ROLL can list several folders (e.g. + data/processed/rollouts/push_extra), DUP=0.
# For the 20-command run the text is padded up to MAXC=20 commands (k / KN=20) and the token limit raised to TOK=192 (a marked 20-command
# sentence is 185 tokens); the real commands per episode stay at most 2.
#   NAME=ours_cyl_mem_even MODE=memory GPU=1 setsid nohup scripts/distill_place.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=${NAME:?}; MODE=${MODE:?}; GPU=${GPU:?}; PORT=${PORT:-6111}; ROLL=${ROLL:-data/processed/rollouts/ours_cyl_fixed_done}; SHARDS=${SHARDS:-12}; DUP=${DUP:-1}; MAXC=${MAXC:-8}; KN=${KN:-6}; TOK=${TOK:-96}
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
log "$NAME: building $SHARDS shards (prompt_mode $MODE, pad even up to $MAXC commands, k/$KN, dup_push $DUP, done channel) from $ROLL"
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $ROLL --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode $MODE --pad even --dup_push $DUP --max_cmds $MAXC --k_norm $KN --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
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
until [ $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) )) -ge 12000 ]; do sleep 120; done
export TOK
log "$NAME: fine-tuning from smolvla_base, up to 100000 steps (GPU $GPU, token limit $TOK)"
scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME 100000 $GPU > $C/ft_$NAME.log 2>&1 &
sleep 120
PID=$(pgrep -f "lerobot-train.*dataset.repo_id=local/$NAME" | head -1)
[ -n "$PID" ] && setsid nohup python3 scripts/early_stop.py --log $C/ft_$NAME.log --pid $PID --window 6000 --min_gain 0.015 >> $C/early_stop_$NAME.log 2>&1 < /dev/null &
log "$NAME: early-stop watchdog on (window 6000 steps, 1.5% gain; trainer pid ${PID:-none})"
FLAGS=$([ $MODE = marks ] && echo --marks || echo "--memory_done --k_norm $KN") NAME=$NAME PORT=$PORT GPU=$GPU scripts/eval_memory_student.sh
