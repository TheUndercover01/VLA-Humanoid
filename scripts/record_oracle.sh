#!/bin/bash
# Oracle distillation data: scripted-expert rollouts with cameras, 500 successful episodes per combined task.
cd "$(dirname "$0")/.."
for t in c1 c2 c3; do
  scripts/run_isaac.sh /media/storage/ayush/cache/rec_oracle_$t.log sim/record_rollouts.py --headless --device cuda:0 \
    --task $t --policy scripted --episodes 500 --batch 100 --seed 100 --out data/processed/rollouts/oracle_$t
done
echo ORACLE DONE
