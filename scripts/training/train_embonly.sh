#!/bin/bash
#SBATCH --job-name=gritlm
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 999:00:00             # maximum executio n time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive

######################
### Set enviroment ###
######################
cd /home/wychanbu/gritlm/gritlm
source /home/wychanbu/gritlm/.gritvenv/bin/activate
# export WANDB_PROJECT="gritlm"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data/wychanbu/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Training setup
GPUS_PER_NODE=4

LAUNCHER="accelerate launch \
    --config_file /home/wychanbu/gritlm/scripts/configs/config_8gpusfsdp_qwen.yml \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --main_process_port 8000 \
    --machine_rank 0 \
    --role localhost: \
    --tee 1 \
    "

TRAIN_DATA=/data/wychanbu/test_data/ # replace with the directory of your training data

export CMD=" \
    -m training.run \
    --output_dir /data/wychanbu/re_models/Qwen2.5-7B_allnli_dureader_fever_hotppotqa_mrtydi_msmarco_nq_quora_squad_t2ranking_triviaqa_hardneg/ \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --train_data $TRAIN_DATA \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 32 \
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
    --attn_implementation sdpa \
    --save_steps 200 \
    --max_grad_norm 1.0 \
    --lora
    "

clear; $LAUNCHER $CMD