#!/bin/bash
# 14 Oct (user: "the VLA does the checking of whether the cube is on the cylinder ... check there"): at every one of the student's own marks,
# where is the object really? (eval_chains --done_diag now logs gap_z_cm = height above its resting pose, off_xy_cm, hand_z_cm, grip).
# ours_cyl_marks_handleg_rand, unstack + restack + swap + c2, 100 val layouts, GPU 0. Status lines MARKPOS; table in cache/markpos_table.txt.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
NAME=ours_cyl_marks_handleg_rand; G=0; PORT=6204; C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data; STATUS=$C/distill_status.txt
log() { echo "$(date '+%a %H:%M') MARKPOS $*" | tee -a $STATUS; }
(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model --port $PORT > $C/server_markpos.log 2>&1) &
S=$!
until grep -q "serving" $C/server_markpos.log 2>/dev/null || ! kill -0 $S 2>/dev/null; do sleep 5; done
for ch in c2 swap unstack restack; do
  T=markpos_$ch
  CUDA_VISIBLE_DEVICES=$G timeout -k 30 2400 scripts/run_isaac.sh $C/ev_$T.log sim/eval_chains.py --headless --device cuda:0 --keypoints \
    --hold_fix --chain $ch --val --policy vla --vla_port $PORT --replan 1 --learned_done --marks --whole_prompt --done_diag --name $T --out runs/eval/$T.csv
  pkill -9 -f "sim/eval_chains.py.*runs/eval/$T.csv" 2>/dev/null
  log "$NAME $ch: $(grep -aoE 'lenient [0-9]+/[0-9]+, in-order [0-9]+/[0-9]+' $C/ev_$T.log | tail -1)"
done
kill $S
python3 - > $C/markpos_table.txt <<'EOF2'
import csv, statistics as st
for ch in ("c2", "swap", "unstack", "restack"):
    rows = [r for r in csv.DictReader(open(f"runs/eval/markpos_{ch}.csv.diag.csv")) if int(r["ok"]) >= 0 and r["skill"] == "place-down"]
    for k in sorted({int(r["command"]) for r in rows}):
        g = [r for r in rows if int(r["command"]) == k]
        for ok in (1, 0):
            h = [r for r in g if int(r["ok"]) == ok]
            if h:
                z = sorted(float(r["gap_z_cm"]) for r in h); xy = sorted(float(r["off_xy_cm"]) for r in h)
                print(f"{ch:8s} put-down #{k + 1} mark {'REAL ' if ok else 'FALSE'} n={len(h):3d} | object above its resting pose: median {st.median(z):5.1f} cm "
                      f"(p10 {z[len(z) // 10]:5.1f}, p90 {z[9 * len(z) // 10]:5.1f}) | off the spot: median {st.median(xy):4.1f} cm | "
                      f"gripper opening {st.median(float(r['grip']) for r in h):.2f} | hand z {st.median(float(r['hand_z_cm']) for r in h):.1f} cm")
EOF2
log "$NAME done (table: $C/markpos_table.txt)"
