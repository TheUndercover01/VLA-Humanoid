#!/bin/bash
# Evaluate one fine-tuned VLA on the validation layouts (seed 4321, never the eval states): picks the
# replanning interval on C1 from {1, 3, 10}, then runs C1 and C2 with it. Used to choose the student model.
#   setsid nohup scripts/eval_vla_val.sh <train_name> [server_gpu] [port] > /dev/null 2>&1 &
# Server: lerobot env on server_gpu; Isaac with cameras on cuda:0 (the only GPU it renders on).
cd "$(dirname "$0")/.."
name=$1; gpu=${2:-1}; port=${3:-6051}
C=/media/storage/ayush/cache
D=/media/storage/ayush/vla_data
STATUS=$C/student_status.txt
log() { echo "$(date '+%a %H:%M') $*" | tee -a $STATUS; }
success() { grep -aoE "on $2: success [0-9]+" "$1" | grep -oE "[0-9]+$" | tail -1; }

(export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD
 exec /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server \
   --policy $D/train/$name/checkpoints/last/pretrained_model --port $port > $C/server_$name.log 2>&1) &
SERVER=$!
until grep -q "serving" $C/server_$name.log 2>/dev/null || ! kill -0 $SERVER 2>/dev/null; do sleep 5; done
kill -0 $SERVER 2>/dev/null || { log "$name: server failed, see $C/server_$name.log"; exit 1; }

run() {   # task replan -> success count
  local tag=${name}_$1_val_r$2
  scripts/run_isaac.sh $C/eval_$tag.log sim/eval.py --headless --enable_cameras --device cuda:0 --task $1 --val \
    --policy vla --vla_port $port --replan $2 --name $name --out runs/eval/$tag.csv
  local s=$(success $C/eval_$tag.log $1); log "$tag: ${s:-?}/100"; echo ${s:-0}
}

best=3; best_s=-1
for r in 1 3 10; do
  s=$(run c1 $r | tail -1)
  [ "$s" -gt "$best_s" ] && { best=$r; best_s=$s; }
done
log "$name: replan interval on C1 validation: $best ($best_s/100)"
run c2 $best > /dev/null
kill $SERVER
log "$name: done"
