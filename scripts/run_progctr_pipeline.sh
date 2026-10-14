#!/bin/bash
# 14 Oct evening (user): drop the pauses; back to "(in progress)" + text contrast at every handoff, so the done flag has to read
# where it is in the sentence. Same cliptol recordings as ours_cyl_marks_cliptol (no new recording), GPU 1.
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_FIXED_TARGET="0.03,-0.12" VLA_CUBE_PUSH_ONLY=1
R=/media/storage/ayush/vla_data/rollouts/ours_cyl_marks_cliptol
export CHAINS="unstack restack restack8 lift c1 c2 swap push lift_cyl c1_cyl rand8_1 rand8_2"
PROGRESS=1 CONTRAST=5 EVAL_EXTRA="--stack_ztol 0.0134" NAME=ours_cyl_marks_progctr MODE=marks GPU=1 ISAAC_GPU=1 PORT=6198 \
  ROLL="${R}_a ${R}_b ${R}_c" SHARDS=24 DUP=0 MAXC=10 KN=10 TOK=128 scripts/distill_place.sh
