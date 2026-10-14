#!/bin/bash
# 10 Oct: HANDOFF task C on the student of run_handleg_pipeline.sh, started when that pipeline logs "pipeline done":
# the teacher's length curve (GPU 1, no lock), then on the student: perturbation study, videos (16 layouts for restack so a 6/6 run
# can exist, 8 for the rest), length curve. Student sims take cache/isaac_eval.lock on GPU 0, servers run on GPU 1.
#   setsid nohup scripts/run_handleg_after.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt; REF=data/processed/keypoint_ref_real_straight.npz
log() { echo "$(date '+%a %H:%M') HANDLEG-C $*" | tee -a $STATUS; }
until L=$(grep -aE "HANDLEG ours_cyl_marks_(handleg|clean): pipeline done" $STATUS | tail -1); [ -n "$L" ]; do
  pgrep -f run_handleg_pipeline.sh > /dev/null || { log "pipeline gone without 'pipeline done', not starting"; exit 1; }
  sleep 120
done
NAME=$(echo "$L" | grep -oE "ours_cyl_marks_(handleg|clean)")
log "task C on $NAME"
if [ $NAME = ours_cyl_marks_handleg ]; then
  T=runs/rl/keypoints_cyl_handleg_nobank_s1/final_teacher.pt
  (for Ln in 4 8 12 16 20; do for s in 1 0; do
     ch=rand${Ln}_$s
     CUDA_VISIBLE_DEVICES=1 timeout -k 30 3600 scripts/run_isaac.sh $C/tev_handleg_len_$ch.log sim/eval_chains.py --headless --device cuda:0 \
       --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs 50 --policy rsl:$T --hand_leg \
       --name handleg_len --out runs/eval/teacher_handleg_len_$ch.csv
     log "teacher handleg_len [$ch, 50 layouts, real template]: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/tev_handleg_len_$ch.log | tail -1)"
   done; done) &
fi
NAME=$NAME GPU=1 PORT=6171 scripts/perturb_study.sh
NAME=$NAME GPU=1 PORT=6172 NENV=16 CHAINS=restack scripts/headline_videos.sh
NAME=$NAME GPU=1 PORT=6173 NENV=8 CHAINS="c2 unstack c3 c1" scripts/headline_videos.sh
NAME=$NAME FLAGS=--marks GPU=1 PORT=6174 scripts/length_curve.sh
wait
log "task C on $NAME done"
