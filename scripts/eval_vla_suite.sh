#!/bin/bash
# Evaluate a served SmolVLA on every task, with the full prompt and with the planner (B1+LLM).
#   scripts/eval_vla_suite.sh <name> <port> [tasks...]
name=$1; port=$2; shift 2
tasks=${@:-lift push c1 c2 c3}
cd "$(dirname "$0")/.."
for t in $tasks; do
  scripts/run_isaac.sh /media/storage/ayush/cache/eval_${name}_$t.log sim/eval.py --headless --enable_cameras \
    --device cuda:0 --task $t --policy vla --vla_port $port --name $name
  scripts/run_isaac.sh /media/storage/ayush/cache/eval_${name}_llm_$t.log sim/eval.py --headless --enable_cameras \
    --device cuda:0 --task $t --policy vla --vla_port $port --plan --name ${name}_llm
done
echo SUITE DONE
