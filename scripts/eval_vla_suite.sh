#!/bin/bash
# Evaluate a served SmolVLA on tasks, with the full prompt and (PLAN=1) with the planner (B1+LLM).
#   [PLAN=0] [VOCAB=data/processed/vocab_standin.pt] scripts/eval_vla_suite.sh <name> <port> [tasks...]
# VOCAB: the model outputs vocabulary actions ([skill one-hot | z]), so evaluate it in the vocabulary env.
name=$1; port=$2; shift 2
tasks=${@:-lift push c1 c2 c3}
plan=${PLAN:-1}
vocab_args=${VOCAB:+--vocab $VOCAB}
cd "$(dirname "$0")/.."
for t in $tasks; do
  scripts/run_isaac.sh /media/storage/ayush/cache/eval_${name}_$t.log sim/eval.py --headless --enable_cameras \
    --device cuda:0 --task $t --policy vla --vla_port $port --name $name $vocab_args
  if [ "$plan" = "1" ]; then
    scripts/run_isaac.sh /media/storage/ayush/cache/eval_${name}_llm_$t.log sim/eval.py --headless --enable_cameras \
      --device cuda:0 --task $t --policy vla --vla_port $port --plan --name ${name}_llm $vocab_args
  fi
done
echo SUITE DONE
