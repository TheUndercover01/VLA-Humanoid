#!/bin/bash
# 10 Oct (user: "make it a three-way thing ... RL can start anywhere ... train everything from scratch"): the two-object pipeline redone with
# the hand's leg (gripper -> object, then the object's leg object -> target; sim/envs/keypoint_env.py --hand_leg) and NO scripted mid-skill
# states (--bank_p 0: 90% of episodes at home, 10% from a stacked tower), in training and in recording. Runs unattended:
#   1. GPU 0: teacher keypoints_cyl_handleg_nobank_s1 = train_teacher.sh + --hand_leg --bank_p 0, 1000 iterations (runs to the end).
#      GPU 1 meanwhile (task B): the old teacher on all 10 chains x 100 layouts and the length curve, real-clip template.
#   2. Gate (HANDOFF task E): new teacher on the same 10 chains (both GPUs); pass = every training skill >= 90% (mean of the last 10
#      iterations) and the mean lenient score over the 10 chains within 5 points of the old teacher's.
#   3. Student from scratch with the recipe of ours_cyl_marks (3000 paired + 1500 push episodes, marks, pad even up to 20, TOK 192,
#      smolvla_base + early stop, 10 chains x 100) via distill_place.sh. Pass -> ours_cyl_marks_handleg on the new teacher.
#      Fail -> ours_cyl_marks_clean on the old teacher (HANDOFF task A), so a clean headline exists either way. Recording uses
#      --bank_p 0 for both teachers (no scripted states in the student's data).
#   setsid nohup scripts/run_handleg_pipeline.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
RUN=keypoints_cyl_handleg_nobank_s1; ITERS=${ITERS:-1000}
OLD=runs/rl/keypoints_cyl_fixedtgt_c3half_filt095_s1/final_teacher.pt
REF=data/processed/keypoint_ref_real_straight.npz
CHAINS="push lift lift_cyl c1 c1_cyl c2 c3 swap unstack restack"
C=/media/storage/ayush/cache; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') HANDLEG $*" | tee -a $STATUS; }
free() { echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) )); }
score() { grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+ \(mean commands in order [0-9.]+/[0-9]+\)' $1 | tail -1; }
teval() {   # teval GPU TAG CKPT CHAIN NENVS EXTRA...: one teacher eval (privileged state, real template)
  local G=$1 tag=$2 ck=$3 ch=$4 ne=$5; shift 5
  CUDA_VISIBLE_DEVICES=$G timeout -k 30 3600 scripts/run_isaac.sh $C/tev_${tag}_$ch.log sim/eval_chains.py --headless --device cuda:0 \
    --keypoints --hold_fix --ref $REF --action_filter 0.95 --chain $ch --val --num_envs $ne --policy rsl:$ck "$@" \
    --name $tag --out runs/eval/teacher_${tag}_$ch.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/teacher_${tag}_$ch.csv" 2>/dev/null   # an Isaac process can hang at shutdown
  log "teacher $tag [$ch, $ne layouts, real template]: $(score $C/tev_${tag}_$ch.log)"
}

# ---------- 1. teacher (GPU 0) + task B (GPU 1) ----------
if [ ! -f runs/rl/$RUN/model_$((ITERS-1)).pt ]; then
  RES=$(ls runs/rl/$RUN/model_*.pt 2>/dev/null | sort -V | tail -1)   # a crashed run continues from its last checkpoint
  log "training $RUN${RES:+ (resumed from $(basename $RES))}: hand leg, no scripted states (bank_p 0), $ITERS iterations x 16384 envs (GPU 0)"
  CUDA_VISIBLE_DEVICES=0 scripts/run_isaac.sh $C/train_$RUN.log sim/train_rl.py --headless --device cuda:0 \
    --task keypoints --seed 1 --num_envs 16384 --max_iterations $ITERS --run_name $RUN --max_chain 2 --hold_fix \
    --action_filter 0.95 --episode_s 20 --save_interval 20 --ref $REF --bank data/processed/state_bank_standin.npz \
    --hand_leg --bank_p 0 ${RES:+--resume $RES} &
  PT=$!
fi
[ -n "$SKIPB" ] || (for ch in $CHAINS; do teval 1 old_real $OLD $ch 100; done
 for L in 4 8 12 16 20; do for s in 1 0; do teval 1 old_real_len $OLD rand${L}_$s 50; done; done
 log "task B (old teacher, real template) done") &
PB=$!
[ -n "$PT" ] && wait $PT
CK=$(ls runs/rl/$RUN/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
[ "$CK" -ge $((ITERS-20)) ] || { log "teacher ended early at model_$CK, stopping (see $C/train_$RUN.log)"; exit 1; }
cp runs/rl/$RUN/model_$CK.pt runs/rl/$RUN/final_teacher.pt
NEW=runs/rl/$RUN/final_teacher.pt
SK=$(python3 - $C/train_$RUN.log <<'EOF'
import re, sys
t = open(sys.argv[1], errors="ignore").read()
out = []
for k in ("push", "pick-lift", "place-down"):
    v = [float(x) for x in re.findall(rf"Skill/{k}: ([0-9.]+)", t)][-10:]
    out.append(f"{k}={sum(v) / max(len(v), 1):.3f}")
print(" ".join(out))
EOF
)
log "teacher $RUN model_$CK done; training skills (mean of the last 10 iterations): $SK"

# ---------- 2. gate: new teacher on the 10 chains (both GPUs) ----------
A="push lift_cyl c1_cyl c3 unstack"; B="lift c1 c2 swap restack"
(for ch in $A; do teval 0 handleg $NEW $ch 100 --hand_leg; done) &
P0=$!
[ -n "$SKIPB" ] || wait $PB
(for ch in $B; do teval 1 handleg $NEW $ch 100 --hand_leg; done) &
P1=$!
wait $P0 $P1
GATE=$(python3 - "$SK" $C "$CHAINS" <<'EOF'
import re, sys
sk, c, chains = sys.argv[1], sys.argv[2], sys.argv[3].split()
def len_(tag, ch):
    try:
        m = re.findall(r"lenient ([0-9]+)/([0-9]+)", open(f"{c}/tev_{tag}_{ch}.log", errors="ignore").read())
        return 100 * int(m[-1][0]) / int(m[-1][1])
    except Exception:
        return None
new = [len_("handleg", ch) for ch in chains]; old = [len_("old_real", ch) for ch in chains]
skills_ok = all(float(v.split("=")[1]) >= 0.9 for v in sk.split())
ok = None not in new and None not in old
d = (sum(new) - sum(old)) / len(chains) if ok else float("nan")
print(("PASS" if ok and skills_ok and d >= -5 else "FAIL") + f" (skills >= 0.9: {skills_ok}; mean lenient new - old over 10 chains: {d:+.1f})")
EOF
)
log "gate: $GATE"

# ---------- 3. student from scratch ----------
if [[ $GATE == PASS* ]]; then
  NAME=ours_cyl_marks_handleg; T=$NEW; TF="--hand_leg"
else
  NAME=ours_cyl_marks_clean; T=$OLD; TF=""
fi
RA=data/processed/rollouts/${NAME}_main_a; RB=data/processed/rollouts/${NAME}_main_b; RP=data/processed/rollouts/${NAME}_push
rec() {   # rec GPU OUT EPISODES SEED
  until [ $(free $1) -ge 10000 ]; do sleep 20; done
  local L=$C/rec_$(basename $2).log
  CUDA_VISIBLE_DEVICES=$1 timeout -k 60 5400 scripts/run_isaac.sh $L sim/record_keypoint_rollouts.py --headless --device cuda:0 \
    --policy rsl:$T $TF --bank_p 0 --hold_fix --ref $REF --action_filter 0.95 --episode_s 20 --paired --episodes $3 --seed $4 --out $2
  log "$NAME recorded $(basename $2): $(grep -aE 'successful episodes|Traceback|OutOfMemory' $L | tail -1 | cut -c1-120)"
}
log "$NAME: recording with $T (bank_p 0): 3000 paired (1500 GPU 1 seed 100, 1500 GPU 0 seed 101) + 1500 push (GPU 0, seed 777)"
rm -rf $RA $RB $RP
rec 1 $RA 1500 100 &
P1=$!
(VLA_PUSH_SHARE=1 rec 0 $RP 1500 777; rec 0 $RB 1500 101) &
P2=$!
wait $P1 $P2
for r in $RA $RB $RP; do grep -aq "successful episodes" $C/rec_$(basename $r).log || { log "$NAME: recording $(basename $r) failed"; exit 1; }; done
NAME=$NAME MODE=marks GPU=0 PORT=6191 ROLL="$RA $RB $RP" SHARDS=24 DUP=0 MAXC=20 KN=20 TOK=192 scripts/distill_place.sh
log "$NAME: pipeline done (teacher $T)"
