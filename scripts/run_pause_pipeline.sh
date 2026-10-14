#!/bin/bash
# 14 Oct (user: "ok do both"): the done flag in ours_cyl_marks_cliptol acts as a "cube lifted and held still" detector (8-command videos: after the
# pick-up mark the robot hovers, the flag stays ~1, 3-5 marks are written in 1-2 s, the skipped commands are never attempted). Two training-side fixes,
# same teacher (keypoints_cyl_cliptol_s1, no fine-tune), every episode still 2 commands:
#   1. PAUSES in the recording (--pause_p 0.012, 5-15 ticks of zero arm motion, the label stays the unpaused teacher's action): frames of a held-still
#      object that are NOT done get done = 0
#   2. CONTRAST at every handoff (build_dataset --contrast 5): each command's 5 done frames again with its (done) already in the sentence and done = 0
# then the student ours_cyl_marks_pause (marks, pad 10), evals at the clips' put-down tolerance, a done audit and 8 videos of restack8. Status: PAUSE.
#   setsid nohup scripts/run_pause_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
NAME=ours_cyl_marks_pause; RI=1.0; PT=0.0134
T=runs/rl/keypoints_cyl_cliptol_s1/final_teacher.pt
REF=data/processed/keypoint_ref_real_straight.npz
TF="--place_tol $PT --cube_push_only"
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
PY=/media/storage/ayush/miniconda3/envs/lerobot/bin/python
log() { echo "$(date '+%a %H:%M') PAUSE $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }

RA=data/processed/rollouts/${NAME}_a; RB=data/processed/rollouts/${NAME}_b; RC=data/processed/rollouts/${NAME}_c
rec() {   # rec GPU OUT SEED
  until [ $(free $1) -ge 10000 ]; do sleep 30; done
  local L=$C/rec_$(basename $2).log
  CUDA_VISIBLE_DEVICES=$1 timeout -k 60 9000 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T --hand_leg --rand_init $RI --bank_p 0 $TF --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes 1500 --seed $3 \
    --pause_p 0.012 --pause_ticks 5,15 --out $2
  log "$NAME recorded $(basename $2): $(grep -aE 'paused ticks|successful episodes|Traceback|OutOfMemory' $L | tail -2 | tr '\n' ' ' | cut -c1-200)"
}
rm -rf $RA $RB $RC
log "$NAME: recording 3 x 1500 with pauses (teacher keypoints_cyl_cliptol_s1, tolerance $PT)"
rec 0 $RA 500 &
P1=$!
(sleep 120; rec 1 $RC 502) &
P2=$!
wait $P1
rec 0 $RB 501
wait $P2
for r in $RA $RB $RC; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "$NAME: recording $(basename $r) failed"; exit 1; }; done

# ---------- student: contrast frames (distill_place.sh builds with its own flags: CONTRAST=5 adds --contrast) ----------
export CHAINS="push lift lift_cyl c1 c1_cyl c2 swap unstack restack restack8 rand8_1 rand8_2"
CONTRAST=5 EVAL_EXTRA="--stack_ztol $PT" NAME=$NAME MODE=marks GPU=0 ISAAC_GPU=0 PORT=6195 ROLL="$RA $RB $RC" SHARDS=24 DUP=0 MAXC=10 KN=10 TOK=128 scripts/distill_place.sh

# ---------- done audit + videos (GPU 0) ----------
[ -d $D/train/$NAME/checkpoints/last/pretrained_model ] || { log "$NAME: no checkpoint"; exit 1; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD
 exec $PY -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port 6196 > $C/server_pause_audit.log 2>&1) &
S=$!
until grep -q "serving" $C/server_pause_audit.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for ch in unstack restack8; do
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 3600 scripts/run_isaac.sh $C/ev_pause_audit_$ch.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port 6196 --replan 1 --learned_done --marks --whole_prompt --stack_ztol $PT --done_diag \
    --name pause_audit_$ch --out runs/eval/pause_audit_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/pause_audit_$ch.csv" 2>/dev/null
  log "$NAME audit $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+' $C/ev_pause_audit_$ch.log | tail -1) | $(grep -aoE 'done flag audit: .*' $C/ev_pause_audit_$ch.log | tail -1 | cut -c1-200)"
done
kill $S
NAME=$NAME FLAGS="--marks --stack_ztol $PT" PORT=6197 GPU=0 CHAINS="restack8" NENV=8 scripts/headline_videos.sh
log "$NAME: pipeline done (videos: media/headline/${NAME}_restack8_env*.mp4)"
