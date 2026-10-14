#!/bin/bash
# 14 Oct (user: tighten the teacher's done check to the eval's standard; then: no fine-tune, use the recent no-lift teacher as it is): the put-down done check accepted the object up to ~2.1 cm above the
# stack (mean corner distance < 2.14 cm, 3 ticks); the eval counts it only within 1.5 cm, held 5 ticks. The recorder ends every episode when the
# teacher's check passes, so the "done" labels sat on frames the eval rejects. The no-lift teacher runs in an env whose
# put-down done check is the eval's standard (--place_tol 0.015 --place_hold 5), so a command (and the recording) ends only then, cube pushes only (user: no "push the cylinder"); then a check,
# 3 x 1500 recorded episodes (labels now on frames that pass the eval), student ours_cyl_marks_strict (marks, pad 10), evals incl. restack (6)
# and restack8 (8 commands).  Status lines STRICT.   setsid nohup scripts/run_strict_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0 VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
RUN=keypoints_cyl_strict_s1; NAME=ours_cyl_marks_strict; RI=1.0
INIT_T=runs/rl/keypoints_cyl_nolift_s1/model_999.pt; ITERS=1400
REF=data/processed/keypoint_ref_real_straight.npz
TF="--place_tol 0.015 --place_hold 5 --cube_push_only"
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') STRICT $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }

# ---------- 1. teacher: the no-lift teacher AS IT IS (user: no fine-tune); the strict done check is only in its env ----------
T=runs/rl/keypoints_cyl_nolift_s1/final_teacher.pt
SK="push=0.993 pick-lift=0.988 place-down=0.986"      # its own training (last 10 iterations), see NOLIFT in the status log
log "teacher: keypoints_cyl_nolift_s1 as it is; put-down done in the recording env within 1.5 cm, held 5 ticks; cube pushes only"

# ---------- 2. check (strict scoring, the teacher in its own strict env) ----------
for ch in c1 c2 swap unstack restack restack8 push lift; do
  until [ $(free 0) -ge 5000 ]; do sleep 30; done
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 3600 scripts/run_isaac.sh $C/tev_strict_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs 100 --policy rsl:$T --hand_leg --place_tol 0.015 --place_hold 5 \
    --name strict --out runs/eval/teacher_strict_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/teacher_strict_$ch.csv" 2>/dev/null
  log "teacher strict [$ch, 100 layouts, strict scoring]: $(score $C/tev_strict_$ch.log)"
done
RS=$(grep -aoE 'in-order [0-9]+/100' $C/tev_strict_restack.log | tail -1 | grep -oE '[0-9]+' | head -1)
OK=$(python3 -c "s='$SK'; print(all(float(v.split('=')[1]) >= 0.9 for v in s.split()) and int('${RS:-0}') >= 85)")
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
rec 0 $RA 300 &
P1=$!
(sleep 120; rec 1 $RC 302) &
P2=$!
wait $P1
rec 0 $RB 301
wait $P2
for r in $RA $RB $RC; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "$NAME: recording $(basename $r) failed"; exit 1; }; done
log "$NAME labels: $(python3 - $RA $RB $RC <<'EOF3'
import glob, sys, collections
import numpy as np
z = collections.defaultdict(list); push_cyl = 0
for d in sys.argv[1:]:
    for f in glob.glob(d + "/ep_*.npz"):
        e = np.load(f, allow_pickle=True)
        push_cyl += "push the cylinder" in str(e["commands"])
        if str(e["prompts"][-1]).startswith("put"):
            z[str(e["prompts"][-1])].append(e["state"][-5:, 2].mean())
print("; ".join(f"{k}: hand z on the done frames {100 * np.mean(v):.1f} cm (n={len(v)})" for k, v in sorted(z.items())) + f"; episodes pushing the cylinder {push_cyl}")
EOF3
)"

# ---------- 4 + 5. student + evals ----------
export CHAINS="push lift lift_cyl c1 c1_cyl c2 swap unstack restack restack8 rand8_1 rand8_2"
NAME=$NAME MODE=marks GPU=0 ISAAC_GPU=0 PORT=6194 ROLL="$RA $RB $RC" SHARDS=24 DUP=0 MAXC=10 KN=10 TOK=128 scripts/distill_place.sh
log "$NAME: pipeline done (teacher $T)"
