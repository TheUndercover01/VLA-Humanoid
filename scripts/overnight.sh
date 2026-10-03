#!/bin/bash
# Overnight chain (3-4 Oct): RL experts -> eval -> distillation rollouts -> SmolVLA fine-tunes (B2, Ours, Oracle) -> eval.
# Runs detached; progress in $STATUS. Each step waits for its inputs, so it can be (re)started at any point.
cd "$(dirname "$0")/.."
C=/media/storage/ayush/cache
D=/media/storage/ayush/vla_data
STATUS=$C/overnight_status.txt
VOCAB=data/processed/vocab_standin.pt
STEPS=10000                           # identical fine-tuning budget for B2, Ours and Oracle
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
wait_exit() { until grep -aq "^EXIT" "$1" 2>/dev/null; do sleep 60; done; }
latest() { ls $1/model_*.pt | sort -V | tail -1; }
success() { grep -aoE "on $2: success [0-9]+" "$1" | grep -oE "[0-9]+$" | tail -1; }

eval_expert() {   # task action -> eval on the 100 eval states with video; echoes success count
  local task=$1 action=$2 dir=runs/rl/${1}_${2}_s1 name=rl_${2}_${1}
  local extra=""; [ "$action" = vocab ] && extra="--vocab $VOCAB"
  scripts/run_isaac.sh $C/eval_$name.log sim/eval.py --headless --enable_cameras --device cuda:0 --task $task \
    --policy rsl:$(latest $dir) $extra --name rl_$action --video media/rl_${action}_${task}_final.mp4 \
    --out runs/eval/rl_${action}_$task.csv
  local s=$(success $C/eval_$name.log $task); log "eval $name: ${s:-?}/100 ($(latest $dir))"; echo ${s:-0}
}

record() {        # task action out -> 500 successful rollouts with cameras
  local task=$1 action=$2 out=$3 extra=""
  [ "$action" = vocab ] && extra="--vocab $VOCAB"
  scripts/run_isaac.sh $C/rec_${action}_$task.log sim/record_rollouts.py --headless --device cuda:0 --task $task \
    --policy rsl:$(latest runs/rl/${task}_${action}_s1) $extra --episodes 500 --batch 100 --seed 100 --out $out
  log "recorded $out: $(grep -aE 'kept' $C/rec_${action}_$task.log | tail -1)"
}

build() {         # name fps src... -> LeRobot dataset
  local name=$1 fps=$2; shift 2
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD /media/storage/ayush/miniconda3/envs/lerobot/bin/python \
    -m vla.build_dataset --src "$@" --repo_id local/$name --root $D/lerobot/$name --fps $fps > $C/build_$name.log 2>&1
  log "built dataset $name: $(grep -aE '^wrote' $C/build_$name.log | tail -1)"
}

finetune() {      # name gpu extra... -> SmolVLA checkpoint
  local name=$1 gpu=$2; shift 2
  scripts/finetune_smolvla.sh $D/lerobot/$name $D/train/$name $STEPS $gpu "$@" > $C/ft_$name.log 2>&1
  log "fine-tuned $name: $(grep -aE '^EXIT' $C/ft_$name.log)"
}

serve_eval() {    # name port gpu tasks... -> eval suite through vla/server.py
  local name=$1 port=$2 gpu=$3; shift 3
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD
   /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server \
     --policy $D/train/$name/checkpoints/last/pretrained_model --port $port > $C/server_$name.log 2>&1) &
  local pid=$!
  until grep -q "serving" $C/server_$name.log 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  PLAN=0 VOCAB="$VOCAB_EVAL" scripts/eval_vla_suite.sh $name $port "$@" > $C/suite_$name.log 2>&1
  pkill -f "vla.server --policy $D/train/$name/"
  for t in "$@"; do log "VLA $name on $t: $(success $C/eval_${name}_$t.log $t)/100"; done
}

log "overnight start"

# --- Oracle: matched to the tasks B2/Ours have tonight (c1, c2) ---
oracle() {
  build oracle_c12 10 data/processed/rollouts/oracle_c1 data/processed/rollouts/oracle_c2
  finetune oracle_c12 0
}

# --- RL experts -> eval -> rollouts (only experts that work) ---
experts() {
  for job in "c2 raw" "c1 raw" "c1 vocab" "c2 vocab"; do
    set -- $job
    wait_exit $C/rl_$1_$2_s1.log
    s=$(eval_expert $1 $2 | tail -1)
    if [ "${s:-0}" -ge 40 ]; then
      if [ "$2" = raw ]; then record $1 raw data/processed/rollouts/b2_$1; else record $1 vocab data/processed/rollouts/ours_$1; fi
    else
      log "skip rollouts for $1 $2: success ${s:-?} < 40"
    fi
  done
}

oracle &
experts
wait
log "experts and oracle done"

# --- B2 and Ours datasets + fine-tunes (in parallel, one per GPU) ---
b2_src=$(ls -d data/processed/rollouts/b2_c1 data/processed/rollouts/b2_c2 2>/dev/null)
ours_src=$(ls -d data/processed/rollouts/ours_c1 data/processed/rollouts/ours_c2 2>/dev/null)
[ -n "$b2_src" ] && (build b2 10 $b2_src; finetune b2 0) &
[ -n "$ours_src" ] && (build ours 1 $ours_src; finetune ours 1 --policy.chunk_size=4 --policy.n_action_steps=1) &
wait

# --- evaluate the distilled VLAs on c1, c2 ---
serve_eval oracle_c12 6021 1 c1 c2
[ -n "$b2_src" ] && serve_eval b2 6022 1 c1 c2
[ -n "$ours_src" ] && VOCAB_EVAL=$VOCAB serve_eval ours 6023 1 c1 c2
log "overnight done"
