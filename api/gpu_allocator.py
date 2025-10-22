"""Automatic GPU allocation based on model size and memory requirements."""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Precision types and their memory footprint (bytes per parameter)
PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Training type memory multipliers
TRAINING_MULTIPLIERS = {
    "sft": 1.0,  # Supervised fine-tuning
    "dpo": 2.0,  # Direct preference optimization
    "ppo": 3.0,  # Proximal policy optimization
    "rlhf": 2.0,  # Reinforcement learning from human feedback
}


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    parameters_billions: float
    family: str
    context_length: int = 2048
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration affecting memory usage."""

    batch_size: int = 1
    sequence_length: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 64
    precision: str = "bf16"
    training_type: str = "sft"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True


# Constants
GPU_MEMORY_CAPACITIES = {"L40S": 48, "A100-80GB": 80, "H100": 80, "T4": 16, "A10G": 24}
SAFETY_MARGIN = 1.2
LORA_OVERHEAD_BASE = 4096
OPTIMIZER_MULTIPLIER = 2.0

# Model registry with known parameter counts
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # LLaMA family
    "meta-llama/Llama-3.2-1B": ModelInfo(
        "meta-llama/Llama-3.2-1B", 1.23, "llama", 128000
    ),
    "meta-llama/Llama-3.2-3B": ModelInfo(
        "meta-llama/Llama-3.2-3B", 3.2, "llama", 128000
    ),
    "meta-llama/Llama-3.1-8B": ModelInfo(
        "meta-llama/Llama-3.1-8B", 8.0, "llama", 128000
    ),
    "meta-llama/Llama-3.1-70B": ModelInfo(
        "meta-llama/Llama-3.1-70B", 70.0, "llama", 128000
    ),
    "meta-llama/Llama-2-7B": ModelInfo("meta-llama/Llama-2-7B", 7.0, "llama", 4096),
    "meta-llama/Llama-2-13B": ModelInfo("meta-llama/Llama-2-13B", 13.0, "llama", 4096),
    "meta-llama/Llama-2-70B": ModelInfo("meta-llama/Llama-2-70B", 70.0, "llama", 4096),
    # Qwen family
    "Qwen/Qwen2.5-1.5B": ModelInfo("Qwen/Qwen2.5-1.5B", 1.5, "qwen", 32768),
    "Qwen/Qwen2.5-3B": ModelInfo("Qwen/Qwen2.5-3B", 3.0, "qwen", 32768),
    "Qwen/Qwen2.5-7B": ModelInfo("Qwen/Qwen2.5-7B", 7.0, "qwen", 32768),
    "Qwen/Qwen2.5-14B": ModelInfo("Qwen/Qwen2.5-14B", 14.0, "qwen", 32768),
    "Qwen/Qwen2.5-32B": ModelInfo("Qwen/Qwen2.5-32B", 32.0, "qwen", 32768),
    "Qwen/Qwen2.5-72B": ModelInfo("Qwen/Qwen2.5-72B", 72.0, "qwen", 32768),
    # Gemma family
    "google/gemma-2-2b": ModelInfo("google/gemma-2-2b", 2.0, "gemma", 8192),
    "google/gemma-2-9b": ModelInfo("google/gemma-2-9b", 9.0, "gemma", 8192),
    "google/gemma-2-27b": ModelInfo("google/gemma-2-27b", 27.0, "gemma", 8192),
    "google/gemma-7b": ModelInfo("google/gemma-7b", 7.0, "gemma", 8192),
    # GLM family
    "THUDM/glm-4-9b": ModelInfo("THUDM/glm-4-9b", 9.0, "glm", 65536),
    "THUDM/glm-4-9b-chat": ModelInfo("THUDM/glm-4-9b-chat", 9.0, "glm", 65536),
    "THUDM/glm-4-9b-chat-1m": ModelInfo("THUDM/glm-4-9b-chat-1m", 9.0, "glm", 1048576),
}


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get model information from registry with efficient lookup."""
    # Direct lookup first (most common case)
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Partial match lookup (only if direct lookup fails)
    model_lower = model_name.lower()
    for key, info in MODEL_REGISTRY.items():
        if model_lower in key.lower() or key.lower() in model_lower:
            logger.info(f"Found model {model_name} via partial match: {key}")
            return info

    logger.warning(f"Model {model_name} not found in registry")
    return None


def calculate_memory_requirements(
    model_info: ModelInfo, config: TrainingConfig
) -> Tuple[float, Dict[str, float]]:
    """Calculate total GPU memory requirements for training."""
    params_billions = model_info.parameters_billions
    bytes_per_param = PRECISION_BYTES[config.precision]
    training_multiplier = TRAINING_MULTIPLIERS[config.training_type]

    # Calculate memory components
    lora_overhead = config.lora_rank / LORA_OVERHEAD_BASE
    model_memory_gb = params_billions * bytes_per_param * (1 + lora_overhead)
    optimizer_memory_gb = params_billions * bytes_per_param * OPTIMIZER_MULTIPLIER

    # Activation memory (simplified calculation)
    estimated_layers = max(24, int(params_billions * 2))
    activation_memory_gb = (
        config.batch_size
        * config.sequence_length
        * estimated_layers
        * bytes_per_param
        / (1024**3)
    )

    # Batch memory
    batch_memory_gb = (
        config.batch_size * config.sequence_length * bytes_per_param / (1024**3)
    )

    # Apply training type multiplier and safety margin
    base_memory = (
        model_memory_gb + optimizer_memory_gb + activation_memory_gb + batch_memory_gb
    )
    total_memory_gb = base_memory * training_multiplier * SAFETY_MARGIN

    return total_memory_gb, {
        "model_memory_gb": model_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "batch_memory_gb": batch_memory_gb,
        "training_multiplier": training_multiplier,
        "total_memory_gb": total_memory_gb,
    }


def allocate_gpu_config(
    model_name: str,
    user_override: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
    **kwargs,
) -> str:
    """Allocate GPU configuration based on comprehensive memory requirements."""
    if user_override:
        logger.info(f"Using user-specified GPU config: {user_override}")
        return user_override

    model_info = get_model_info(model_name)
    if model_info is None:
        logger.warning(f"Unknown model {model_name}, using conservative allocation")
        return "L40S:1"  # Use supported GPU type

    config = training_config or TrainingConfig()
    total_memory_gb, breakdown = calculate_memory_requirements(model_info, config)

    logger.info(f"Memory requirements for {model_name}: {total_memory_gb:.1f}GB")
    gpu_config = _find_optimal_gpu_config(total_memory_gb)

    logger.info(
        f"Auto-allocated {gpu_config} for {model_name} ({model_info.parameters_billions}B params)"
    )
    return gpu_config


def _find_optimal_gpu_config(required_memory_gb: float) -> str:
    """Find optimal GPU configuration based on memory requirements."""
    # Only use GPU types that are supported by Modal training sessions
    supported_gpus = {
        "L40S": 48,
        "A100-80GB": 80,
        "H100": 80,
    }
    
    # Try single GPU first (sorted by memory capacity)
    for gpu_type, capacity in sorted(supported_gpus.items(), key=lambda x: x[1]):
        if capacity >= required_memory_gb:
            return f"{gpu_type}:1"

    # Multi-GPU allocation based on memory requirements
    a100_capacity = supported_gpus["A100-80GB"]
    h100_capacity = supported_gpus["H100"]

    if required_memory_gb <= a100_capacity * 2:
        return "A100-80GB:2"
    elif required_memory_gb <= a100_capacity * 4:
        return "A100-80GB:4"
    elif required_memory_gb <= a100_capacity * 8:
        return "A100-80GB:8"
    elif required_memory_gb <= h100_capacity * 4:
        return "H100:4"
    else:
        return "H100:8"


def get_supported_models() -> Dict[str, ModelInfo]:
    """Get all supported models in the registry."""
    return MODEL_REGISTRY.copy()


def estimate_training_memory(
    model_name: str,
    batch_size: int = 1,
    sequence_length: int = 2048,
    lora_rank: int = 32,
    precision: str = "bf16",
    training_type: str = "sft",
) -> Tuple[float, Dict[str, float]]:
    """Estimate memory requirements for training a specific model."""
    model_info = get_model_info(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} not found in registry")

    config = TrainingConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        lora_rank=lora_rank,
        precision=precision.lower(),
        training_type=training_type.lower(),
    )

    return calculate_memory_requirements(model_info, config)


# Predefined training configurations for different efficiency levels
EFFICIENCY_CONFIGS = [
    {"lora_rank": 16, "precision": "int4", "sequence_length": 2048},  # High efficiency
    {"lora_rank": 32, "precision": "int8", "sequence_length": 2048},  # Balanced
    {"lora_rank": 32, "precision": "bf16", "sequence_length": 2048},  # Standard
    {"lora_rank": 64, "precision": "fp16", "sequence_length": 2048},  # High quality
]


def suggest_training_config(
    model_name: str, available_memory_gb: float, training_type: str = "sft"
) -> Dict[str, Any]:
    """Suggest optimal training configuration for a model given memory constraints."""
    model_info = get_model_info(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} not found in registry")

    training_type = training_type.lower()

    # Try configurations from most efficient to least efficient
    for config_params in EFFICIENCY_CONFIGS:
        config = TrainingConfig(
            batch_size=1,
            sequence_length=config_params["sequence_length"],
            lora_rank=config_params["lora_rank"],
            precision=config_params["precision"],
            training_type=training_type,
            gradient_checkpointing=True,
        )

        memory_gb, breakdown = calculate_memory_requirements(model_info, config)
        if memory_gb <= available_memory_gb:
            return {
                "batch_size": config.batch_size,
                "sequence_length": config.sequence_length,
                "lora_rank": config.lora_rank,
                "precision": config.precision,
                "training_type": config.training_type,
                "gradient_checkpointing": config.gradient_checkpointing,
                "estimated_memory_gb": memory_gb,
                "memory_breakdown": breakdown,
                "fits_in_memory": True,
            }

    # If no configuration fits, return the most efficient one with warning
    most_efficient = EFFICIENCY_CONFIGS[0]
    config = TrainingConfig(
        batch_size=1,
        sequence_length=most_efficient["sequence_length"],
        lora_rank=most_efficient["lora_rank"],
        precision=most_efficient["precision"],
        training_type=training_type,
        gradient_checkpointing=True,
    )

    memory_gb, breakdown = calculate_memory_requirements(model_info, config)
    return {
        "batch_size": config.batch_size,
        "sequence_length": config.sequence_length,
        "lora_rank": config.lora_rank,
        "precision": config.precision,
        "training_type": config.training_type,
        "gradient_checkpointing": config.gradient_checkpointing,
        "estimated_memory_gb": memory_gb,
        "memory_breakdown": breakdown,
        "fits_in_memory": False,
        "warning": f"Model requires {memory_gb:.1f}GB but only {available_memory_gb}GB available",
    }
