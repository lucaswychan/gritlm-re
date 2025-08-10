#!/bin/bash
#SBATCH --job-name=gritlm
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 999:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive

######################
### Set enviroment ###
######################
cd /home/wychanbu/gritlm/gritlm
source /home/wychanbu/gritlm/.gritvenv/bin/activate
# export WANDB_PROJECT="gritlm"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3,5
export HF_HOME=/data/wychanbu/huggingface
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Training setup
GPUS_PER_NODE=2

LAUNCHER="accelerate launch \
    --config_file /home/wychanbu/gritlm/scripts/configs/config_8gpusfsdp_qwen.yml \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --main_process_port 8000 \
    --machine_rank 0 \
    --role localhost: \
    --tee 1 \
    "

TRAIN_DATA=/data/wychanbu/re_data/hard-neg # replace with the directory of your training data

export CMD=" \
    -m training.run \
    --output_dir /data/wychanbu/re_models/Qwen3-0.6B_hard_neg_no_lora_cosine_0p1wd/ \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --train_data $TRAIN_DATA \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last \
    --normalized \
    --temperature 0.02 \
    --train_group_size 4 \
    --negatives_cross_device \
    --query_max_len 512 \
    --passage_max_len 512 \
    --mode embedding \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --use_unique_indices \
    --attn bbcc \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --save_steps 200 \
    "

clear; $LAUNCHER $CMD