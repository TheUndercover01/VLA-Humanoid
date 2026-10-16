#!/bin/bash
# 14 Oct (user: loose everywhere): the no-lift RL teacher (privileged state) scored with the same loose check as the final student
# (scripts/eval_cliptol_loose.sh). Waits for a GPU with >= 8 GB free.  G=1 setsid nohup scripts/eval_teacher_loose.sh &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
T=runs/rl/keypoints_cyl_nolift_s1/final_teacher.pt; REF=data/processed/keypoint_ref_real_straight.npz; G=${G:-1}
LOOSE="--xy_tol 0.05 --stack_ztol 0.03 --target_z 0.06 --lift_rise 0.04 --lenient_hold 2 --ordered_hold 2"
C=/media/storage/ayush/cache
log() { echo "$(date '+%a %H:%M') LOOSE teacher nolift $*" | tee -a $C/distill_status.txt; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $G) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G) )); }
for ch in ${CHAINS:-restack unstack restack8 rand8_1 rand8_2 c3 lift c1 c2 swap push lift_cyl c1_cyl}; do
  until [ $(free) -ge 8000 ]; do sleep 60; done
  CUDA_VISIBLE_DEVICES=$G timeout -k 30 3600 scripts/run_isaac.sh $C/tev_loose_nolift_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs 100 --policy rsl:$T --hand_leg $LOOSE \
    --name teacher_loose_nolift --out runs/eval/teacher_loose_nolift_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*teacher_loose_nolift_$ch.csv" 2>/dev/null
  log "$ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/tev_loose_nolift_$ch.log | tail -1)"
done
