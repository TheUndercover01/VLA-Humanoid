#!/bin/bash
# Fine-tune SmolVLA on a local LeRobot dataset (lerobot env). Front/wrist cameras feed the base model's camera1/camera2.
#   scripts/finetune_smolvla.sh <dataset_root> <output_dir> <steps> [gpu] [extra lerobot-train args...]
root=$1; out=$2; steps=$3; gpu=${4:-0}; shift 4
export HF_HOME=/media/storage/ayush/cache/hf HF_LEROBOT_HOME=/media/storage/ayush/vla_data/lerobot
export CUDA_VISIBLE_DEVICES=$gpu NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1   # accelerate refuses P2P on RTX 4090s
/media/storage/ayush/miniconda3/envs/lerobot/bin/lerobot-train --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/$(basename "$root") --dataset.root="$root" \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=32 --steps="$steps" --log_freq=100 --save_freq=2000 --num_workers=4 \
  --output_dir="$out" --job_name="$(basename "$out")" --wandb.enable=false "$@"
echo "EXIT $?"
