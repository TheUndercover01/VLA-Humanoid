#!/bin/bash
# 10 Oct: why does ours_cyl_marks_handleg (teacher 99-100 on every short task) score push 16, unstack 40 / 4 in order, restack 6 / 0?
# (1) videos of push + unstack (8 layouts each, headline_videos.sh), (2) done-flag audit (eval_chains --done_diag, default flag 0.85 / 5 ticks,
# the same audit as DONESTUDY on 8 Oct) on handleg and, for a same-day comparison, on realref. All sim jobs take the shared isaac_eval.lock.
#   setsid nohup scripts/diagnose_handleg.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
GPU=${GPU:-0}; C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
log "DIAG handleg: start (videos push + unstack, then the done-flag audit)"
NAME=ours_cyl_marks_handleg CHAINS="push unstack" PORT=6211 GPU=$GPU scripts/headline_videos.sh
serve() {   # name port
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
   exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$1/checkpoints/last/pretrained_model --port $2 > $C/server_diag_$1.log 2>&1) &
  SERVER=$!
  until grep -q "serving" $C/server_diag_$1.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
  kill -0 $SERVER 2>/dev/null || { log "DIAG $1: server failed"; exit 1; }
}
audit() {   # name port chain envs
  exec 9>$C/isaac_eval.lock; flock 9
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 2400 scripts/run_isaac.sh $C/diag_$1_$3.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $3 --val --num_envs $4 --policy vla --vla_port $2 --replan 1 --learned_done --marks --whole_prompt \
    --done_thresh 0.85 --done_hold 5 --done_vote 1 --done_diag --name $1_diag --out runs/eval/diag_$1_$3.csv
  flock -u 9; exec 9>&-
  pkill -9 -f "sim/eval_chains.py.*diag_$1_$3.csv" 2>/dev/null
  log "DIAG $1 $3 ($4 layouts): $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $C/diag_$1_$3.log | tail -1) | $(grep -aoE 'done flag audit: .*' $C/diag_$1_$3.log | tail -1)"
}
serve ours_cyl_marks_handleg 6212
for ch in unstack restack push c2; do audit ours_cyl_marks_handleg 6212 $ch 100; done
kill $SERVER
serve ours_cyl_marks_realref 6213
audit ours_cyl_marks_realref 6213 unstack 100
kill $SERVER
log "DIAG handleg: done"
