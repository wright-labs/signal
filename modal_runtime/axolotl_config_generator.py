"""Axolotl configuration generator for multi-GPU training.

Converts user RunConfig parameters to Axolotl YAML format.
"""
import yaml
from typing import Dict, Any, Optional


def generate_axolotl_config(
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[list] = None,
    learning_rate: float = 3e-4,
    max_seq_length: int = 2048,
    optimizer: str = "adamw_8bit",
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    num_gpus: int = 1,
    output_dir: str = "/data/outputs",
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
) -> str:
    """Generate Axolotl YAML config from run parameters.
    
    Args:
        base_model: HuggingFace model ID
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Modules to apply LoRA to (None = auto)
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        optimizer: Optimizer type
        bf16: Use bfloat16 precision
        gradient_checkpointing: Enable gradient checkpointing
        num_gpus: Number of GPUs (1 for single, >1 for multi-GPU)
        output_dir: Output directory for checkpoints
        micro_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        YAML configuration string for Axolotl
    """
    # Base configuration
    config = {
        "base_model": base_model,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        # Quantization: 8-bit for single GPU, none for multi-GPU (FSDP incompatible)
        "load_in_8bit": num_gpus == 1,
        "load_in_4bit": False,
        # LoRA configuration
        "adapter": "lora",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_linear": True if not lora_target_modules else False,
        "lora_target_modules": lora_target_modules if lora_target_modules else None,
        # Sequence and packing
        "sequence_len": max_seq_length,
        "sample_packing": False,
        # Training configuration
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": 1,  # We control steps via API, not epochs
        "optimizer": _convert_optimizer_name(optimizer),
        "learning_rate": learning_rate,
        "lr_scheduler": "constant",  # No scheduler, user controls LR via API
        # Precision and efficiency
        "bf16": "auto" if bf16 else False,
        "tf32": False,
        "gradient_checkpointing": gradient_checkpointing,
        "flash_attention": True,
        # Output
        "output_dir": output_dir,
        "logging_steps": 1,
        "saves_per_epoch": 0,  # Disable auto-saves, we control via API
        # Special tokens
        "special_tokens": {
            "pad_token": "<|end_of_text|>",
        },
    }
    
    # Multi-GPU specific configuration
    if num_gpus > 1:
        config.update({
            "fsdp": [
                "full_shard",
                "auto_wrap",
            ],
            "fsdp_config": {
                "fsdp_transformer_layer_cls_to_wrap": _get_transformer_layer_class(base_model),
                "fsdp_state_dict_type": "FULL_STATE_DICT",
                "fsdp_offload_params": False,
                "fsdp_sync_module_states": True,
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            },
        })
    
    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}
    
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def _convert_optimizer_name(optimizer: str) -> str:
    """Convert Signal optimizer name to Axolotl optimizer name."""
    optimizer_map = {
        "adamw_8bit": "adamw_bnb_8bit",
        "adamw": "adamw_torch",
        "adam": "adam",
        "sgd": "sgd",
    }
    return optimizer_map.get(optimizer, "adamw_bnb_8bit")


def _get_transformer_layer_class(base_model: str) -> str:
    """Get the transformer layer class name for FSDP wrapping.
    
    Args:
        base_model: HuggingFace model ID
        
    Returns:
        Class name for FSDP auto-wrapping
    """
    # Map model families to their transformer layer classes
    model_lower = base_model.lower()
    
    if "llama" in model_lower:
        return "LlamaDecoderLayer"
    elif "mistral" in model_lower:
        return "MistralDecoderLayer"
    elif "qwen" in model_lower:
        return "Qwen2DecoderLayer"
    elif "gemma" in model_lower:
        return "GemmaDecoderLayer"
    elif "phi" in model_lower:
        return "PhiDecoderLayer"
    else:
        # Default to Llama (most common)
        return "LlamaDecoderLayer"


def write_config_to_volume(
    config_yaml: str,
    config_path: str,
    volume=None,
) -> Dict[str, Any]:
    """Write YAML configuration to disk and optionally commit to volume.
    
    Args:
        config_yaml: YAML configuration string
        config_path: Path where config should be written
        volume: Optional Modal Volume to commit
        
    Returns:
        Parsed configuration dictionary
    """
    import os
    
    config_dict = yaml.safe_load(config_yaml)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write config
    with open(config_path, "w") as f:
        f.write(config_yaml)
    
    # Commit volume if provided
    if volume is not None:
        volume.commit()
    
    return config_dict

