# GritLM Training Speed Optimization Guide

This document outlines the comprehensive optimizations implemented to increase GritLM training speed and efficiency.

## Overview of Optimizations

The optimizations focus on several key areas:
1. **Data Processing & Loading**
2. **Model Forward Pass**
3. **Memory Efficiency**
4. **Compute Optimizations**
5. **Training Infrastructure**

## 1. Data Processing & Loading Optimizations

### Data Preprocessing (`run.py`)
- **String Pre-processing**: Pre-compiled static prompt parts to reduce string concatenation overhead
- **Efficient Filtering**: Added character-based length estimation before expensive tokenization
- **Optimized Multiprocessing**: Increased worker processes for large datasets (up to 8 workers)

### Dataset Processing (`data.py`)
- **Reduced Memory Allocations**: Optimized string truncation with conditional checks
- **Vectorized Operations**: Improved negative sampling with list comprehensions
- **Efficient Label Creation**: Optimized generative label masking with vectorized operations
- **Removed Training Loop Logging**: Eliminated expensive logging calls during training

### Data Loading
- **Optimized DataLoader Settings**: 
  - `num_workers=4` (adaptive based on dataset size)
  - `pin_memory=True` for faster GPU transfer
  - `persistent_workers=True` to avoid worker respawn overhead
  - `prefetch_factor=2` for better pipelining

## 2. Model Forward Pass Optimizations

### Loss Computation (`model.py`)
- **Removed Training Loop Logging**: Eliminated expensive logging calls in contrastive loss
- **Optimized Attention Mask Operations**: Reduced unnecessary tensor cloning
- **Efficient Loss Combination**: Replaced `sum()` with direct addition for better performance
- **Vectorized Instruction Masking**: Improved batch processing for instruction token masking

### Memory Optimizations
- **In-place Operations**: Used in-place operations where safe to reduce memory allocations
- **Optimized Tensor Operations**: Reduced intermediate tensor creation

## 3. Memory Efficiency Improvements

### Gradient Checkpointing (`run.py`)
- **Optimized Checkpointing Settings**: 
  - `use_reentrant=False` for better memory efficiency
  - `preserve_rng_state=False` for faster execution (safe in most cases)

### Mixed Precision Training
- **Automatic dtype Selection**: Uses optimal dtype based on training arguments (fp16/bf16/fp32)
- **Proper Mixed Precision Setup**: Ensures compatibility with hardware capabilities

## 4. Compute Optimizations

### PyTorch Compile (`optimizations.py`)
- **Accelerate Integration**: Uses accelerate's TorchDynamoPlugin for seamless torch.compile integration
- **Regional Compilation**: Automatically enables regional compilation for distributed training (faster compilation)
- **Full Model Compilation**: Uses full model compilation for single-GPU training (better optimization)
- **Backend Optimization**: Uses "inductor" backend with optimized settings
- **Conditional Compilation**: Automatically disabled for QLoRA and when torch.compile is unavailable

### Optimized Attention Backends
- **Flash Attention**: Enabled `flash_sdp` for memory-efficient attention
- **Memory Efficient Attention**: Enabled `mem_efficient_sdp` as fallback
- **Math SDP**: Enabled optimized math kernels

### CUDA Optimizations
- **Memory Management**: Optimized CUDA memory fraction allocation
- **Kernel Optimizations**: Enabled optimized CUDA kernels where available

## 5. Training Infrastructure Optimizations

### Custom Training Arguments (`arguments.py`)
- **Performance Defaults**: Added optimized default values for data loading
- **Automatic Configuration**: Post-init hook to set optimal values

### Comprehensive Optimization Module (`optimizations.py`)
- **Centralized Optimizations**: Single module to apply all performance improvements
- **Adaptive Settings**: Automatically adjusts settings based on dataset size and hardware
- **Performance Monitoring**: Logs system information for debugging

## Usage

The optimizations are automatically applied when using the training script. Key settings:

```python
# Training arguments for optimal performance
training_args = CustomTrainingArguments(
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    bf16=True,  # or fp16 based on hardware
    torch_compile_mode="reduce-overhead",  # Options: "default", "reduce-overhead", "max-autotune"
    # ... other args
)
```

## Expected Performance Improvements

1. **Data Loading**: 20-40% faster due to optimized workers and prefetching
2. **Model Forward Pass**: 15-25% faster due to reduced logging and optimized operations
3. **Memory Usage**: 10-20% reduction due to gradient checkpointing and in-place operations
4. **Overall Training**: 25-50% faster training throughput depending on model size and hardware

## Hardware-Specific Recommendations

### For A100/H100 GPUs:
- Enable BF16 mixed precision
- Use Flash Attention
- Set higher batch sizes with gradient accumulation

### For V100/RTX GPUs:
- Use FP16 mixed precision
- Enable memory-efficient attention
- Optimize batch size for memory constraints

### CPU Optimizations:
- Increase dataloader workers (up to CPU cores)
- Enable persistent workers
- Use optimized tokenizers

## Monitoring Performance

Use the built-in performance logging:

```python
from .optimizations import log_performance_info
log_performance_info()
```

This will show:
- PyTorch version and CUDA info
- GPU specifications
- Available optimizations
- Current settings

## Troubleshooting

### Common Issues:
1. **OOM Errors**: Reduce batch size or enable gradient checkpointing
2. **Compilation Errors**: Disable torch.compile for complex models
3. **Worker Errors**: Reduce num_workers if experiencing deadlocks
4. **Accelerate + torch.compile Issues**: 
   - Error: `KeyError: '_orig_mod'` when using accelerate with compiled models
   - Solution: Now uses accelerate's built-in TorchDynamoPlugin for seamless integration
   - Regional compilation is automatically used for distributed training to avoid conflicts

### Performance Monitoring:
- Monitor GPU utilization (should be >85%)
- Check dataloader efficiency (minimize data loading wait time)
- Verify mixed precision is working (check tensor dtypes)

## Future Optimizations

Potential areas for further improvement:
1. **Custom CUDA Kernels**: For specialized operations
2. **Model Parallelism**: For very large models
3. **Async Data Loading**: For complex preprocessing pipelines
4. **Dynamic Batching**: For variable sequence lengths

## Benchmarking

To measure the impact of optimizations:

1. **Before**: Run training with original code
2. **After**: Run with optimizations enabled
3. **Compare**: 
   - Samples/second
   - Memory usage
   - Time per epoch
   - GPU utilization

The optimizations should provide significant improvements across all metrics.