"""
Performance optimizations for GritLM training.
This module contains various optimizations to improve training speed and memory efficiency.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def enable_optimized_attention():
    """Enable optimized attention backends for faster training."""
    try:
        # Enable optimized CUDA kernels
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Enabled optimized CUDA attention backends")
    except Exception as e:
        logger.warning(f"Could not enable optimized attention: {e}")



def optimize_cuda_settings():
    """Optimize CUDA settings for better performance."""
    if torch.cuda.is_available():
        # Enable optimized memory allocator
        try:
            torch.cuda.empty_cache()
            # Set memory fraction to allow for better memory management
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.95)
            logger.info("Optimized CUDA memory settings")
        except Exception as e:
            logger.warning(f"Could not optimize CUDA settings: {e}")


def create_torch_dynamo_plugin(mode: str = "default", use_regional_compilation: bool = None):
    """
    Create a TorchDynamoPlugin for accelerate's built-in torch.compile integration.
    
    Args:
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        use_regional_compilation: Whether to use regional compilation (auto-detected if None)
    
    Returns:
        TorchDynamoPlugin instance
    """
    try:
        from accelerate.utils import TorchDynamoPlugin
    except ImportError:
        logger.warning("TorchDynamoPlugin not available - update accelerate for torch.compile support")
        return None
    
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",  # Most optimized backend
        mode=mode,
        fullgraph=False,  # More stable than fullgraph=True
        dynamic=False,     # Static shapes for better optimization
        use_regional_compilation=True
    )
    
    compilation_type = "regional" if use_regional_compilation else "full model"
    logger.info(f"Created TorchDynamoPlugin with {compilation_type} compilation (mode: {mode})")
    
    return dynamo_plugin

def optimize_tokenizer_settings(tokenizer):
    """Optimize tokenizer settings for better performance."""
    # Enable fast tokenizer if available
    if hasattr(tokenizer, 'is_fast') and tokenizer.is_fast:
        # Fast tokenizers are already optimized
        logger.info("Using fast tokenizer")
    else:
        logger.warning("Slow tokenizer detected - consider using a fast tokenizer for better performance")
    
    return tokenizer

def get_optimal_dataloader_settings(dataset_size: int, batch_size: int) -> Dict[str, Any]:
    """
    Get optimal dataloader settings based on dataset size and batch size.
    
    Args:
        dataset_size: Size of the dataset
        batch_size: Batch size
        
    Returns:
        Dictionary with optimal dataloader settings
    """
    # Calculate optimal number of workers
    num_workers = min(4, max(1, torch.get_num_threads() // 2))
    
    # For small datasets, reduce workers to avoid overhead
    if dataset_size < 1000:
        num_workers = 1
    elif dataset_size < 10000:
        num_workers = 2
    
    settings = {
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    logger.info(f"Optimal dataloader settings: {settings}")
    return settings

def optimize_model_for_training(model: nn.Module) -> nn.Module:
    """
    Apply various optimizations to the model for training.
    
    Args:
        model: The model to optimize
        
    Returns:
        Optimized model
    """
    # Set model to training mode
    model.train()
    
    # Enable optimized operations
    if hasattr(model, 'config'):
        # Disable unnecessary caching
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    
    # Optimize for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        # This is handled in the main training script
        pass
    
    logger.info("Applied model training optimizations")
    return model

def setup_mixed_precision(training_args) -> bool:
    """
    Setup mixed precision training optimally.
    
    Args:
        training_args: Training arguments
        
    Returns:
        Whether mixed precision is enabled
    """
    if training_args.fp16:
        # Ensure FP16 is properly configured
        if not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA not available")
            return False
        logger.info("FP16 mixed precision enabled")
        return True
    elif training_args.bf16:
        # Check if BF16 is supported
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logger.info("BF16 mixed precision enabled")
            return True
        else:
            logger.warning("BF16 requested but not supported on this device")
            return False
    
    return False

def log_performance_info():
    """Log performance-related information."""
    logger.info("=== Performance Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CPU threads: {torch.get_num_threads()}")
    logger.info(f"Torch compile available: {hasattr(torch, 'compile')}")
    logger.info("================================")

def apply_all_optimizations(model, tokenizer, training_args, dataset_size=None, batch_size=None):
    """
    Apply all available optimizations.
    
    Args:
        model: The model to optimize
        tokenizer: The tokenizer to optimize
        training_args: Training arguments
        dataset_size: Size of dataset (optional)
        batch_size: Batch size (optional)
        
    Returns:
        Tuple of (optimized_model, optimized_tokenizer, dataloader_settings, dynamo_plugin)
    """
    logger.info("Applying performance optimizations...")
    
    # Log system info
    log_performance_info()
    
    # Enable optimized attention
    enable_optimized_attention()
    
    # Optimize CUDA settings
    optimize_cuda_settings()
    
    # Setup mixed precision
    setup_mixed_precision(training_args)
    
    # Optimize model
    model = optimize_model_for_training(model)
    
    # Create TorchDynamoPlugin for accelerate's torch.compile integration
    dynamo_plugin = None
    if not getattr(training_args, 'qlora', False) and hasattr(torch, 'compile'):
        compile_mode = getattr(training_args, 'torch_compile_mode', 'default')
        dynamo_plugin = create_torch_dynamo_plugin(mode=compile_mode)
    
    # Optimize tokenizer
    tokenizer = optimize_tokenizer_settings(tokenizer)
    
    # Get optimal dataloader settings
    dataloader_settings = {}
    if dataset_size and batch_size:
        dataloader_settings = get_optimal_dataloader_settings(dataset_size, batch_size)
    
    logger.info("Performance optimizations applied successfully")
    
    return model, tokenizer, dataloader_settings, dynamo_plugin