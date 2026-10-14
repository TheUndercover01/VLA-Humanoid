#!/bin/bash
# 14 Oct (user: "find the problem in the data set that's causing stack to not work in order"): every recorded put-down was the LAST command of
# its episode (2169 of 2169), so the student never saw what follows a put-down's (done) mark; half the data was push -> pick, a quarter pushed
# the cylinder (topples; user: remove it). Fix = a new episode kind, still 2 commands: the object starts in the hand (held start), "put it down,
# then pick up the next one" (sim/envs/keypoint_env.py held_p). Mix after the paired twins: put -> pick 40%, pick -> put 40%, cube push -> pick 20%.
#   1. teacher keypoints_cyl_handoff_s1: 400-iteration fine-tune of the no-lift teacher (79% on held starts before) with that mix, GPU 0
#   2. check: held-start episodes (sim/test_held_start.py) + c1 / c2 / swap / unstack / restack / restack8 / push / lift, 100 layouts
#   3. record 3 x 1500 episodes, 4. student ours_cyl_marks_handoff from smolvla_base (marks, pad 10, no cylinder push in the padding),
#   5. evals incl. restack (6 commands) and restack8 (8 commands) + rand8 chains.          status lines HANDOFF
#   setsid nohup scripts/run_handoff_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
RUN=keypoints_cyl_handoff_s1; NAME=ours_cyl_marks_handoff; RI=1.0
INIT_T=runs/rl/keypoints_cyl_nolift_s1/model_999.pt; ITERS=1400
REF=data/processed/keypoint_ref_real_straight.npz
MIX="--held_p 0.33 --push_p 0.585 --cube_push_only"
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') HANDOFF $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }

# ---------- 1. teacher fine-tune ----------
if [ ! -f runs/rl/$RUN/model_$((ITERS-1)).pt ]; then
  RES=$(ls runs/rl/$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)
  until [ $(free 0) -ge 14000 ]; do sleep 30; done
  log "teacher $RUN: fine-tune of the no-lift teacher from ${RES:-$INIT_T}, to iteration $ITERS, held starts 33% + cube pushes, GPU 0"
  CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/train_$RUN.log sim/train_rl.py --headless --device cuda:0 \
    --task keypoints --seed 1 --num_envs 16384 --max_iterations $ITERS --run_name $RUN --max_chain 2 --hold_fix \
    --action_filter 0.95 --episode_s 20 --save_interval 20 --ref $REF --bank data/processed/state_bank_standin.npz \
    --hand_leg --push_lift_pen 0.3 --bank_p 0 --rand_init $RI $MIX --resume ${RES:-$INIT_T}
fi
CK=$(ls runs/rl/$RUN/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
[ "$CK" -ge $((ITERS-20)) ] || { log "teacher ended early at model_$CK (see $C/train_$RUN.log)"; exit 1; }
cp runs/rl/$RUN/model_$CK.pt runs/rl/$RUN/final_teacher.pt
T=runs/rl/$RUN/final_teacher.pt
SK=$(python3 - $C/train_$RUN.log <<'EOF2'
import re, sys
t = open(sys.argv[1], errors="ignore").read()
print(" ".join(f"{k}={sum(v) / max(len(v), 1):.3f}" for k in ("push", "pick-lift", "place-down")
               for v in [[float(x) for x in re.findall(rf"Skill/{k}: ([0-9.]+)", t)][-10:]]))
EOF2
)
log "teacher $RUN model_$CK done; training skills (last 10 iterations): $SK"

# ---------- 2. check ----------
CUDA_VISIBLE_DEVICES=0 timeout 900 scripts/run_isaac.sh $C/held_$RUN.log sim/test_held_start.py --headless --device cuda:0 --teacher $T
log "teacher held starts: $(grep -aE '^\[teacher' $C/held_$RUN.log | tail -1)"
for ch in c1 c2 swap unstack restack restack8 push lift; do
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 3600 scripts/run_isaac.sh $C/tev_handoff_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs 100 --policy rsl:$T --hand_leg \
    --name handoff --out runs/eval/teacher_handoff_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/teacher_handoff_$ch.csv" 2>/dev/null
  log "teacher handoff [$ch, 100 layouts]: $(score $C/tev_handoff_$ch.log)"
done
BOTH=$(grep -aoE 'both done [0-9]+/[0-9]+' $C/held_$RUN.log | tail -1 | grep -oE '[0-9]+' | head -1)
OK=$(python3 -c "s='$SK'; print(all(float(v.split('=')[1]) >= 0.9 for v in s.split()) and int('${BOTH:-0}') >= 109)")   # 85% of 128
[ "$OK" = True ] || { log "check failed (skills $SK, held-start both done ${BOTH:-?}/128): no recording"; exit 1; }
log "check passed"

# ---------- 3. record ----------
RA=data/processed/rollouts/${NAME}_a; RB=data/processed/rollouts/${NAME}_b; RC=data/processed/rollouts/${NAME}_c
rec() {   # rec GPU OUT SEED
  until [ $(free $1) -ge 10000 ]; do sleep 30; done
  local L=$C/rec_$(basename $2).log
  CUDA_VISIBLE_DEVICES=$1 timeout -k 60 7200 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T --hand_leg --rand_init $RI --bank_p 0 $MIX --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes 1500 --seed $3 --out $2
  log "$NAME recorded $(basename $2): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $L | tail -1 | cut -c1-120)"
}
rm -rf $RA $RB $RC
log "$NAME: recording 3 x 1500 (GPU 0; c on GPU 1 when it has room)"
rec 0 $RA 200 &
P1=$!
(sleep 120; rec 1 $RC 202) &
P2=$!
wait $P1
rec 0 $RB 201
wait $P2
for r in $RA $RB $RC; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "$NAME: recording $(basename $r) failed"; exit 1; }; done
log "$NAME episode mix: $(python3 - $RA $RB $RC <<'EOF3'
import glob, sys, collections
import numpy as np
c = collections.Counter()
for d in sys.argv[1:]:
    for f in glob.glob(d + "/ep_*.npz"):
        cs = str(np.load(f, allow_pickle=True)["commands"]).split(" > ")
        k = "put->pick" if cs[0].startswith("put") else "push->pick" if cs[0].startswith("push") else "pick->put"
        c[k] += 1
        c["cyl push"] += any(x.startswith("push the cylinder") for x in cs)
n = sum(v for k, v in c.items() if k != "cyl push")
print(", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in c.items()))
EOF3
)"

# ---------- 4 + 5. student + evals (GPU 0) ----------
export CHAINS="push lift lift_cyl c1 c1_cyl c2 swap unstack restack restack8 rand8_1 rand8_2 rand8_3"
NAME=$NAME MODE=marks GPU=0 ISAAC_GPU=0 PORT=6193 ROLL="$RA $RB $RC" SHARDS=24 DUP=0 MAXC=10 KN=10 TOK=128 scripts/distill_place.sh
log "$NAME: pipeline done (teacher $T)"
