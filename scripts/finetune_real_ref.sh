#!/bin/bash
# 9 Oct: the cube + can rollouts behind ours_cyl_marks were recorded in an env that loaded the STAND-IN template (the recorder had no
# --ref until 8 Oct): the teacher (trained on the real-clip template) saw stand-in waypoints and the pick-up goal was the stand-in's
# 14.4 cm lift instead of the clips' 20.4 cm. Here the same teacher is recorded again with the real-clip template (record_keypoint_rollouts
# --ref default), a smaller set (1200 paired + 600 push episodes, same 2:1 mix as before), and the marks student is fine-tuned FROM ITS
# OWN CHECKPOINT on it (short budget), then evaluated on the main chains.
#   setsid nohup scripts/finetune_real_ref.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=ours_cyl_marks_realref; REC_GPU=${REC_GPU:-1}; FT_GPU=${FT_GPU:-0}; STEPS=${STEPS:-8000}; PORT=6171; SHARDS=12
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt; T0=$C/teacher_ours_cyl_fixed_done_gpu0.pt
R1=data/processed/rollouts/realref_main; R2=data/processed/rollouts/realref_push
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
INIT_CK=$D/train/ours_cyl_marks/checkpoints/last/pretrained_model
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
REF=data/processed/keypoint_ref_real_straight.npz
log "$NAME: recording 1200 paired + 600 push episodes with the real-clip template (GPU $REC_GPU)"
CUDA_VISIBLE_DEVICES=$REC_GPU scripts/run_isaac.sh $C/rec_${NAME}_main.log sim/record_keypoint_rollouts.py --headless --device cuda:0 \
  --policy rsl:$T0 --hold_fix --action_filter 0.95 --episode_s 20 --paired --episodes 1200 --ref $REF --seed 2024 --out $R1
log "$NAME main recorded: $(grep -aE 'successful episodes|Traceback' $C/rec_${NAME}_main.log | tail -1)"
VLA_PUSH_SHARE=1 CUDA_VISIBLE_DEVICES=$REC_GPU scripts/run_isaac.sh $C/rec_${NAME}_push.log sim/record_keypoint_rollouts.py --headless --device cuda:0 \
  --policy rsl:$T0 --hold_fix --action_filter 0.95 --episode_s 20 --paired --episodes 600 --ref $REF --seed 2025 --out $R2
log "$NAME push recorded: $(grep -aE 'successful episodes|Traceback' $C/rec_${NAME}_push.log | tail -1)"
# same dataset recipe as ours_cyl_marks (distill_place.sh, MODE=marks, MAXC=20, KN=20, DUP=0, TOK=192)
for i in $(seq 0 $((SHARDS-1))); do
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD $PY -m vla.build_dataset --src $R1 $R2 --repo_id local/${NAME}_s$i \
    --root $D/lerobot/shards/${NAME}_s$i --prompt_mode marks --pad even --dup_push 0 --max_cmds 20 --k_norm 20 --done_frames 5 --streaming --shard $i/$SHARDS > $C/build_${NAME}_s$i.log 2>&1 &
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
until [ $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $FT_GPU) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $FT_GPU) )) -ge 12000 ]; do sleep 60; done
log "$NAME: fine-tuning FROM ours_cyl_marks, up to $STEPS steps (GPU $FT_GPU, token limit 192)"
rm -rf $D/train/$NAME
INIT=$INIT_CK TOK=192 scripts/finetune_smolvla.sh $D/lerobot/$NAME $D/train/$NAME $STEPS $FT_GPU > $C/ft_$NAME.log 2>&1
log "$NAME fine-tuned: $(grep -aoE 'step:[0-9.K]+ .*loss:[0-9.]+' $C/ft_$NAME.log | tail -1)"
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "$NAME: no checkpoint"; exit 1; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$FT_GPU PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_$NAME.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_$NAME.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
for ch in c1 c2 unstack restack lift push; do
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 2400 scripts/run_isaac.sh $C/ev_${NAME}_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt \
    --name $NAME --out runs/eval/${NAME}_$ch.csv
  flock -u 9; exec 9>&-
  pkill -9 -f "sim/eval_chains.py.*runs/eval/${NAME}_$ch.csv" 2>/dev/null
  log "REALREF $NAME $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/ev_${NAME}_$ch.log | tail -1)"
done
kill $SERVER
log "REALREF $NAME done"
