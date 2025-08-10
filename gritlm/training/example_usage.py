#!/usr/bin/env python3
"""
Example usage of optimized GritLM training with accelerate's torch.compile integration.
"""

from gritlm.training.arguments import CustomTrainingArguments
from gritlm.training.optimizations import create_torch_dynamo_plugin

def example_usage():
    """Example of how to use the new accelerate-based torch.compile integration."""
    
    # Create training arguments with torch.compile optimization
    training_args = CustomTrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,  # Use BF16 for better performance on modern GPUs
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        # New torch.compile settings
        torch_compile_mode="reduce-overhead",  # Options: "default", "reduce-overhead", "max-autotune"
        # Other optimized settings
        save_steps=500,
        logging_steps=100,
        eval_steps=500,
    )
    
    # Create TorchDynamoPlugin (this is done automatically in apply_all_optimizations)
    dynamo_plugin = create_torch_dynamo_plugin(
        mode=training_args.torch_compile_mode,
        use_regional_compilation=None  # Auto-detected based on distributed training
    )
    
    print("=== GritLM Training Configuration ===")
    print(f"Torch compile mode: {training_args.torch_compile_mode}")
    print(f"Mixed precision: {'BF16' if training_args.bf16 else 'FP16' if training_args.fp16 else 'FP32'}")
    print(f"Data workers: {training_args.dataloader_num_workers}")
    print(f"Pin memory: {training_args.dataloader_pin_memory}")
    print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
    
    if dynamo_plugin:
        compilation_type = "Regional" if dynamo_plugin.use_regional_compilation else "Full model"
        print(f"Compilation: {compilation_type} ({dynamo_plugin.mode} mode)")
        print(f"Backend: {dynamo_plugin.backend}")
    else:
        print("Compilation: Disabled (torch.compile not available)")
    
    print("\n=== Performance Tips ===")
    print("1. Use 'max-autotune' mode for maximum performance (slower compilation)")
    print("2. Regional compilation is automatically used for distributed training")
    print("3. Ensure your GPU supports the selected mixed precision mode")
    print("4. Monitor GPU utilization to verify optimizations are working")
    
    return training_args, dynamo_plugin

if __name__ == "__main__":
    # Demonstrate usage
    args, plugin = example_usage()
    
    print(f"\nTraining arguments created successfully!")
    print(f"Ready to use with: python -m gritlm.training.run --config_file your_config.json")