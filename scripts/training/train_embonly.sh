#!/bin/bash
# IMPORTANT: Run this script with bash, not sh
# Usage: bash scripts/training/train_embonly.sh
#        NOT: sh scripts/training/train_embonly.sh
#
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
source .venv/bin/activate
cd gritlm

# GPUs 0,1,2,4,5 are occupied by other users' jobs (vLLM servers etc.). Scheduling a rank
# onto a busy GPU makes FSDP init hang forever: that rank cannot allocate the ~15 GiB for
# the unsharded bf16 model, so it never joins the NCCL sync_module_states broadcast while
# the remaining ranks spin at 100% utilization waiting for it.
export CUDA_VISIBLE_DEVICES=1,3,6,7
export GPUS_PER_NODE=4

# Fail fast (instead of hanging in NCCL) if a selected GPU does not have enough free memory.
MIN_FREE_MIB=30000
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    if [ "$free_mib" -lt "$MIN_FREE_MIB" ]; then
        echo "ERROR: GPU $gpu has only ${free_mib} MiB free (< ${MIN_FREE_MIB} MiB needed)." >&2
        echo "Another job is likely using it. Update CUDA_VISIBLE_DEVICES to free GPUs." >&2
        exit 1
    fi
done

export TORCH_CUDA_ARCH_LIST="8.9"
export HF_HOME=/data/wychanbu/hugginface

# Optimized NCCL settings for multi-GPU training
# export NCCL_P2P_DISABLE=1 
# export NCCL_IB_DISABLE=1   # Disable InfiniBand (not available on most setups)
# export NCCL_DEBUG=INFO     # Enable debug output to see what's happening
# export NCCL_TIMEOUT=3600   # Increase timeout to 1 hour
# # export NCCL_BLOCKING_WAIT=1  # Use blocking wait for better error messages
# # Avoid problematic network interfaces
# export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0
export NCCL_P2P_DISABLE=1

# Memory and performance optimizations
# garbage_collection_threshold reclaims cached-but-unused blocks before fragmentation forces an OOM
# (the observed failure reported large reserved-but-unallocated memory).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export TORCH_CUDNN_V8_API_ENABLED=1  # Use cuDNN v8 API

# Disable torch.compile/dynamo (incompatible with FSDP + GradCache)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Enable TF32 for faster matmul on Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1

# Note: Use --report_to none flag instead of WANDB_DISABLED env var (deprecated in wandb v5)

LAUNCHER="accelerate launch \
    --config_file ../scripts/configs/config_8gpusfsdp_qwen.yml \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --main_process_port 8001 \
    --machine_rank 0 \
    --role localhost: \
    --tee 1 \
    "

TRAIN_DATA=/data/wychanbu/re_data/hard-neg # replace with the directory of your training data
TRAIN_DATA=/data/wychanbu/re_data/hard-neg # replace with the directory of your training data

export CMD=" \
    -m training.run \
    --output_dir /data/wychanbu/re_models/qwen3-4b_0p05_sigreg_64bsz \
    --model_name_or_path Qwen/Qwen3-4B \
    --train_data $TRAIN_DATA \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last \
    --normalized \
    --temperature 0.02 \
    --train_group_size 4 \
    --negatives_cross_device \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --attn bb \
    --attn_implementation flash_attention_2 \
    --save_steps 0.25 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --report_to none \
    --sigreg_weight 0.05 \
    "
    # --save_steps accepts an absolute step count or a fraction in (0,1), e.g. 0.25 saves at each quarter
    # --gradient_checkpointing \
# Note: torch.compile is automatically disabled with FSDP due to compatibility issues
# For single-GPU training, you can add: --torch_compile --torch_compile_mode reduce-overhead
# See scripts/training/performance_notes.md for behavior-preserving speed/memory knobs.

clear; $LAUNCHER $CMD