#!/bin/bash
# Overnight chain v2 (4 Oct): the first Oracle VLA (clean demos, 10k steps) got 6/100 on C1 although it fits
# its training data, i.e. it cannot recover from its own errors. v2 records every distillation dataset with
# DART noise (sim/record_rollouts.py --noise 0.3), fine-tunes 20k steps, and picks the replanning interval on
# validation layouts (never the eval states). Same procedure for Oracle, B2 and Ours.
cd "$(dirname "$0")/.."
C=/media/storage/ayush/cache
D=/media/storage/ayush/vla_data
STATUS=$C/overnight_status.txt
VOCAB=data/processed/vocab_standin.pt
STEPS=20000
LOCK=$C/gpu0_record.lock
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
wait_exit() { until grep -aq "^EXIT" "$1" 2>/dev/null; do sleep 60; done; }
latest() { ls $1/model_*.pt | sort -V | tail -1; }
success() { grep -aoE "on $2: success [0-9]+" "$1" | grep -oE "[0-9]+$" | tail -1; }

eval_expert() {   # task action -> eval on the eval states with video; echoes success
  local task=$1 action=$2 dir=runs/rl/${1}_${2}_s1 extra=""
  [ "$action" = vocab ] && extra="--vocab $VOCAB"
  flock $LOCK scripts/run_isaac.sh $C/eval_rl_${action}_$task.log sim/eval.py --headless --enable_cameras --device cuda:0 \
    --task $task --policy rsl:$(latest $dir) $extra --name rl_$action --video media/rl_${action}_${task}_final.mp4 \
    --out runs/eval/rl_${action}_$task.csv
  local s=$(success $C/eval_rl_${action}_$task.log $task); log "eval rl_${action}_$task: ${s:-?}/100 ($(latest $dir))"; echo ${s:-0}
}

record() {        # task policy out [extra] -> 500 successful rollouts with cameras and DART noise
  local task=$1 policy=$2 out=$3; shift 3
  rm -rf $out
  flock $LOCK scripts/run_isaac.sh $C/rec_$(basename $out).log sim/record_rollouts.py --headless --device cuda:0 \
    --task $task --policy $policy "$@" --episodes 500 --batch 100 --seed 100 --noise 0.3 --out $out
  log "recorded $out: $(grep -aE 'kept' $C/rec_$(basename $out).log | tail -1)"
}

build() {         # name fps src...
  local name=$1 fps=$2; shift 2
  HF_HOME=$C/hf HF_LEROBOT_HOME=$D/lerobot PYTHONPATH=$PWD /media/storage/ayush/miniconda3/envs/lerobot/bin/python \
    -m vla.build_dataset --src "$@" --repo_id local/$name --root $D/lerobot/$name --fps $fps > $C/build_$name.log 2>&1
  log "built dataset $name: $(grep -aE '^wrote' $C/build_$name.log | tail -1)"
}

finetune() {      # name gpu extra...
  local name=$1 gpu=$2; shift 2
  scripts/finetune_smolvla.sh $D/lerobot/$name $D/train/$name $STEPS $gpu "$@" > $C/ft_$name.log 2>&1
  log "fine-tuned $name: $(grep -aE '^EXIT' $C/ft_$name.log) $(grep -aoE 'step:[0-9.]+K [^ ]+ [^ ]+ [^ ]+ loss:[0-9.]+' $C/ft_$name.log | tail -1)"
}

serve() {         # name port gpu -> starts the server in the background, waits until it listens
  local name=$1 port=$2 gpu=$3
  (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD
   exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server \
     --policy $D/train/$name/checkpoints/last/pretrained_model --port $port > $C/server_$name.log 2>&1) &
  SERVER_PID=$!
  until grep -q "serving" $C/server_$name.log 2>/dev/null || ! kill -0 $SERVER_PID 2>/dev/null; do sleep 5; done
}

vla_eval() {      # name port task replan [val] [vocab]
  local name=$1 port=$2 task=$3 replan=$4 val=$5 vocab=$6 tag=$1_$3; local extra=""
  [ -n "$val" ] && { extra="--val"; tag=${tag}_val_r$replan; }
  [ -n "$vocab" ] && extra="$extra --vocab $vocab"
  flock $LOCK scripts/run_isaac.sh $C/eval_$tag.log sim/eval.py --headless --enable_cameras --device cuda:0 --task $task \
    --policy vla --vla_port $port --replan $replan --name $name $extra --out runs/eval/${tag}.csv
  local s=$(success $C/eval_$tag.log $task); log "VLA $tag: ${s:-?}/100"; echo ${s:-0}
}

log "overnight v2 start (DART noise 0.3, $STEPS fine-tune steps)"

oracle_branch() {
  record c1 scripted data/processed/rollouts/oracle_dart_c1
  record c2 scripted data/processed/rollouts/oracle_dart_c2
  build oracle_dart 10 data/processed/rollouts/oracle_dart_c1 data/processed/rollouts/oracle_dart_c2
  finetune oracle_dart 0
}

experts_branch() {
  record c2 rsl:$(latest runs/rl/c2_raw_s1) data/processed/rollouts/b2_dart_c2
  wait_exit $C/rl_c1_raw_s1.log
  s=$(eval_expert c1 raw | tail -1)
  [ "${s:-0}" -ge 40 ] && record c1 rsl:$(latest runs/rl/c1_raw_s1) data/processed/rollouts/b2_dart_c1 || log "skip b2 c1 ($s)"
  wait_exit $C/rl_c1_vocab_s1.log
  s=$(eval_expert c1 vocab | tail -1)
  [ "${s:-0}" -ge 40 ] && record c1 rsl:$(latest runs/rl/c1_vocab_s1) data/processed/rollouts/ours_dart_c1 --vocab $VOCAB || log "skip ours c1 ($s)"
  wait_exit $C/rl_c2_vocab_s1.log
  s=$(eval_expert c2 vocab | tail -1)
  if [ "${s:-0}" -lt 40 ]; then           # stacking with 2 s primitives learns slowly: give it 250 more iterations
    log "vocab c2 at ${s:-?}/100 after 250 iterations: continuing to 500"
    scripts/run_isaac.sh $C/rl_c2_vocab_s1_more.log sim/train_rl.py --headless --task c2 --action vocab --seed 1 \
      --max_iterations 500 --device cuda:1 --resume $(latest runs/rl/c2_vocab_s1)
    s=$(eval_expert c2 vocab | tail -1)
  fi
  [ "${s:-0}" -ge 40 ] && record c2 rsl:$(latest runs/rl/c2_vocab_s1) data/processed/rollouts/ours_dart_c2 --vocab $VOCAB || log "skip ours c2 ($s)"
}

oracle_branch &
experts_branch
wait
log "datasets recorded"

b2_src=$(ls -d data/processed/rollouts/b2_dart_c1 data/processed/rollouts/b2_dart_c2 2>/dev/null)
ours_src=$(ls -d data/processed/rollouts/ours_dart_c1 data/processed/rollouts/ours_dart_c2 2>/dev/null)
[ -n "$b2_src" ] && (build b2_dart 10 $b2_src; finetune b2_dart 0) &
[ -n "$ours_src" ] && (build ours_dart 1 $ours_src; finetune ours_dart 1 --policy.chunk_size=4 --policy.n_action_steps=1) &
wait

# replanning interval for raw-action VLAs, chosen with the Oracle VLA on validation layouts
serve oracle_dart 6041 1
best=3; best_s=-1
for r in 1 3 10; do
  s=$(vla_eval oracle_dart 6041 c1 $r val | tail -1)
  [ "${s:-0}" -gt "$best_s" ] && { best=$r; best_s=$s; }
done
log "replan interval chosen on validation layouts: $best"
for t in c1 c2; do vla_eval oracle_dart 6041 $t $best > /dev/null; done
kill $SERVER_PID
if [ -n "$b2_src" ]; then
  serve b2_dart 6042 1; for t in c1 c2; do vla_eval b2_dart 6042 $t $best > /dev/null; done; kill $SERVER_PID
fi
if [ -n "$ours_src" ]; then
  serve ours_dart 6043 1; for t in c1 c2; do vla_eval ours_dart 6043 $t 1 "" $VOCAB > /dev/null; done; kill $SERVER_PID
fi
log "overnight v2 done"
