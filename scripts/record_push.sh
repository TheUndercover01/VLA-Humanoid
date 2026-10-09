#!/bin/bash
# 7 Oct evening: extra push data for the distillation. The same teacher and recording settings as distill_cyl_fixed.sh, every episode a
# fresh-layout push (then pick up), paired (each object commanded on the same layout), half of the pushes from the c3 layouts.
#   setsid nohup scripts/record_push.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12" VLA_PUSH_SHARE=1
EPISODES=${EPISODES:-1500}; ROLL=data/processed/rollouts/push_extra; GPU=${GPU:-1}
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt; T0=$C/teacher_ours_cyl_fixed_done_gpu0.pt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
[ -f $T0 ] || { log "push_extra: teacher copy $T0 missing"; exit 1; }
log "push_extra: recording $EPISODES paired push episodes (GPU $GPU)"
exec 9>$C/isaac_eval.lock; flock 9
CUDA_VISIBLE_DEVICES=$GPU scripts/run_isaac.sh $C/rec_push_extra.log sim/record_keypoint_rollouts.py --headless --device cuda:0 \
  --policy rsl:$T0 --hold_fix --action_filter 0.95 --episode_s 20 --paired --episodes $EPISODES --seed 777 --out $ROLL
flock -u 9; exec 9>&-
log "push_extra recorded: $(grep -aE 'successful episodes|Traceback' $C/rec_push_extra.log | tail -1)"
