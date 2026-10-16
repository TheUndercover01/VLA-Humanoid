#!/bin/bash
# 12 Oct 20:25: resume of run_handleg_rand_pipeline.sh. Its recording of main_a (GPU 1, seed 100) died at start with a CUDA OOM (my z-test evals
# shared GPU 1 with it; the crashed recorder, PID 1784398, then hung at shutdown and still holds 10 GB), so the pipeline stopped at 17:54 BST.
# push (1500) and main_b (1500) are already recorded and are NOT deleted (the pipeline script would rm -rf them). main_a is recorded here on GPU 0
# (same flags and seed), then the student is built, trained and evaluated exactly as in the pipeline (MAXC=10 KN=10 TOK=128).
#   setsid nohup scripts/run_handleg_rand_resume.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
RUN=keypoints_cyl_handleg_rand_s1; RI=1.0; NAME=ours_cyl_marks_handleg_rand; T=runs/rl/$RUN/final_teacher.pt
REF=data/processed/keypoint_ref_real_straight.npz
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
RA=data/processed/rollouts/${NAME}_main_a; RB=data/processed/rollouts/${NAME}_main_b; RP=data/processed/rollouts/${NAME}_push
log() { echo "$(date '+%a %H:%M') HANDLEG_RAND $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
until [ $(free 0) -ge 12000 ]; do sleep 20; done
rm -rf $RA
L=$C/rec_$(basename $RA).log
log "$NAME: re-recording main_a on GPU 0 (1500 paired, seed 100) after the OOM"
CUDA_VISIBLE_DEVICES=0 timeout -k 60 7200 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
  --policy rsl:$T --hand_leg --rand_init $RI --bank_p 0 --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes 1500 --seed 100 --out $RA
log "$NAME recorded $(basename $RA): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $L | tail -1 | cut -c1-120)"
for r in $RA $RB $RP; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "$NAME: recording $(basename $r) failed"; exit 1; }; done
NAME=$NAME MODE=marks GPU=0 PORT=6191 ROLL="$RA $RB $RP" SHARDS=24 DUP=0 MAXC=10 KN=10 TOK=128 scripts/distill_place.sh
log "$NAME: pipeline done (teacher $T)"
