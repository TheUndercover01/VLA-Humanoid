#!/bin/bash
# Fine-tune any LeRobot policy on a local dataset (lerobot env), with the same batch size and logging as
# scripts/finetune_smolvla.sh. The policy is given by the extra args (--policy.path=... or --policy.type=...).
#   scripts/finetune_policy.sh <dataset_root> <output_dir> <steps> <gpu> --policy.path=lerobot/xvla-base [args...]
#   RENAME= scripts/finetune_policy.sh ... --policy.type=multi_task_dit [args...]
# Pretrained checkpoints get front/wrist renamed to camera1/camera2; a policy trained from scratch keeps the
# dataset's names (lerobot refuses a rename without a checkpoint), so pass RENAME= (empty) for those.
root=$1; out=$2; steps=$3; gpu=$4; shift 4
RENAME=${RENAME-'{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'}
export HF_HOME=/media/storage/ayush/cache/hf HF_LEROBOT_HOME=/media/storage/ayush/vla_data/lerobot
export CUDA_VISIBLE_DEVICES=$gpu NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1   # accelerate refuses P2P on RTX 4090s
/media/storage/ayush/miniconda3/envs/lerobot/bin/lerobot-train \
  --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/$(basename "$root") --dataset.root="$root" \
  ${RENAME:+--rename_map="$RENAME"} \
  --batch_size=32 --steps="$steps" --log_freq=100 --save_freq=2000 --num_workers=4 \
  --output_dir="$out" --job_name="$(basename "$out")" --wandb.enable=false "$@"
echo "EXIT $?"
