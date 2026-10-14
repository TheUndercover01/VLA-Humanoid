#!/bin/bash
# 14 Oct (user: "train everything from scratch and infer the put-down thing from the clips, then distill as usual"): the put-down done check
# of the reward gets its tolerance from the real clips (motion/place_tol.py: 95th percentile of how far the placed objects end from their exact
# resting height, 1.21 cm, + the clips' end turn = 1.34 cm; the hand-set value was 2.14 cm). Rest speed / hold time stay hand-set (user).
#   1. teacher keypoints_cyl_cliptol_s1 FROM SCRATCH: the no-lift recipe (hand leg, random start, push-lift penalty 0.3, no scripted starts),
#      cube pushes only (no "push the cylinder"), --place_tol <clips>, 1000 iterations x 16384 envs
#   2. check (strict scoring with the same tolerance), 3. record 3 x 1500 paired, 4. student ours_cyl_marks_cliptol from smolvla_base
#      (marks, pad 10), 5. evals incl. restack (6) and restack8 (8 commands) + rand8, scored with the same put-down tolerance. Status: CLIPTOL.
#   setsid nohup scripts/run_cliptol_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
RUN=keypoints_cyl_cliptol_s1; NAME=ours_cyl_marks_cliptol; RI=1.0; ITERS=${ITERS:-1000}
REF=data/processed/keypoint_ref_real_straight.npz
PT=$(python3 -m motion.place_tol --ref $REF)
TF="--place_tol $PT --cube_push_only"
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') CLIPTOL $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }

# ---------- 1. teacher from scratch ----------
if [ ! -f runs/rl/$RUN/model_$((ITERS-1)).pt ]; then
  RES=$(ls runs/rl/$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)     # a crashed run continues from its last checkpoint
  until [ $(free 0) -ge 11000 ]; do sleep 30; done
  log "training $RUN${RES:+ (resumed from $(basename $RES))} FROM SCRATCH: put-down tolerance from the clips $PT m (was 0.0214), cube pushes only, no-lift recipe, $ITERS iterations x 16384 envs, GPU 0"
  CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/train_$RUN.log sim/train_rl.py --headless --device cuda:0 \
    --task keypoints --seed 1 --num_envs 16384 --max_iterations $ITERS --run_name $RUN --max_chain 2 --hold_fix \
    --action_filter 0.95 --episode_s 20 --save_interval 20 --ref $REF --bank data/processed/state_bank_standin.npz \
    --hand_leg --push_lift_pen 0.3 --bank_p 0 --rand_init $RI $TF ${RES:+--resume $RES}
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
for ch in c1 c2 swap unstack restack restack8 push lift; do
  until [ $(free 0) -ge 5000 ]; do sleep 30; done
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 3600 scripts/run_isaac.sh $C/tev_cliptol_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs 100 --policy rsl:$T --hand_leg --place_tol $PT --stack_ztol $PT \
    --name cliptol --out runs/eval/teacher_cliptol_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/teacher_cliptol_$ch.csv" 2>/dev/null
  log "teacher cliptol [$ch, 100 layouts, put-down tolerance $PT]: $(score $C/tev_cliptol_$ch.log)"
done
RS=$(grep -aoE 'in-order [0-9]+/100' $C/tev_cliptol_restack.log | tail -1 | grep -oE '[0-9]+' | head -1)
OK=$(python3 -c "s='$SK'; print(all(float(v.split('=')[1]) >= 0.9 for v in s.split()) and int('${RS:-0}') >= 80)")
[ "$OK" = True ] || { log "check failed (skills $SK, restack in order ${RS:-?}/100): no recording"; exit 1; }
log "check passed"

# ---------- 3. record ----------
RA=data/processed/rollouts/${NAME}_a; RB=data/processed/rollouts/${NAME}_b; RC=data/processed/rollouts/${NAME}_c
rec() {   # rec GPU OUT SEED
  until [ $(free $1) -ge 10000 ]; do sleep 30; done
  local L=$C/rec_$(basename $2).log
  CUDA_VISIBLE_DEVICES=$1 timeout -k 60 7200 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T --hand_leg --rand_init $RI --bank_p 0 $TF --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes 1500 --seed $3 --out $2
  log "$NAME recorded $(basename $2): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $L | tail -1 | cut -c1-120)"
}
rm -rf $RA $RB $RC
rec 0 $RA 400 &
P1=$!
(sleep 120; rec 1 $RC 402) &
P2=$!
wait $P1
rec 0 $RB 401
wait $P2
for r in $RA $RB $RC; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "$NAME: recording $(basename $r) failed"; exit 1; }; done
log "$NAME labels: $(python3 - $RA $RB $RC <<'EOF3'
import glob, sys, collections
import numpy as np
z = collections.defaultdict(list); push_cyl = 0; n = 0
for d in sys.argv[1:]:
    for f in glob.glob(d + "/ep_*.npz"):
        e = np.load(f, allow_pickle=True); n += 1
        push_cyl += "push the cylinder" in str(e["commands"])
        if str(e["prompts"][-1]).startswith("put"):
            z[str(e["prompts"][-1])].append(e["state"][-5:, 2].mean())
print(f"{n} episodes; " + "; ".join(f"{k}: hand z on the done frames {100 * np.mean(v):.1f} cm (n={len(v)})" for k, v in sorted(z.items())) + f"; pushing the cylinder {push_cyl}")
EOF3
)"

# ---------- 4 + 5. student + evals (same put-down tolerance in the scoring) ----------
export CHAINS="push lift lift_cyl c1 c1_cyl c2 swap unstack restack restack8 rand8_1 rand8_2"
EVAL_EXTRA="--stack_ztol $PT" NAME=$NAME MODE=marks GPU=0 ISAAC_GPU=0 PORT=6194 ROLL="$RA $RB $RC" SHARDS=24 DUP=0 MAXC=10 KN=10 TOK=128 scripts/distill_place.sh
log "$NAME: pipeline done (teacher $T)"
