# Embedding Training Performance Notes

These knobs preserve the training objective and are intended for controlled speed/memory experiments.

## Low-risk flags

- `--use_fused_adamw`: uses Hugging Face's `adamw_torch_fused` optimizer path instead of the default AdamW implementation. Do not combine with `--use_muon`.
- `--gc_chunk_size N`: overrides GradCache's per-forward chunk size. The default is the original `per_device_train_batch_size`, so existing launches are unchanged. Larger chunks reduce FSDP all-gather rounds if VRAM allows.

## Runtime knobs to benchmark

- `GRITLM_GC_NO_SYNC=1`: makes GradCache reduce gradients only on the last chunk. This is mathematically identical, but keeps larger gradient buffers resident.
- `GRITLM_GC_NO_RNG=1`: skips GradCache RNG-state save/restore. Use only when the model path is deterministic, such as dropout-free full finetuning without LoRA dropout.
- `NCCL_P2P_DISABLE`: avoid forcing this to `1` on P2P-capable machines unless it is needed for stability. Disabling P2P can route FSDP traffic through slower paths.
- Gradient checkpointing: GradCache already limits activation residency to one chunk. Disabling `--gradient_checkpointing` can improve speed when the selected `--gc_chunk_size` fits in memory.
- FSDP sharding: `fsdp_sharding_strategy: SHARD_GRAD_OP` can reduce repeated parameter all-gathers at the cost of keeping parameters resident between forward and backward.
- Logging syncs: increase `--logging_steps` and consider `--logging_nan_inf_filter False` for throughput measurements. This removes per-step host/device syncs used only for logging and nan fallback accounting.

## Suggested validation loop

1. Run a short fixed-seed baseline and record step time plus `torch.cuda.max_memory_allocated()`.
2. Change one knob at a time.
3. Keep loss curves identical for Tier 1 changes and within normal floating-point tolerance for runtime/memory knobs.
