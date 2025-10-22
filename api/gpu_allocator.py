"""Automatic GPU allocation based on model size and memory requirements."""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GPUConfigError(ValueError):
    """Raised when GPU config is invalid."""
    pass


VALID_GPU_TYPES = ["L40S", "A100", "A100-80GB", "H100", "T4", "A10G"]


def validate_gpu_config(gpu_config: str, raise_http_exception: bool = False) -> bool:
    """Validate GPU configuration format.
    
    Args:
        gpu_config: GPU config string (e.g., "L40S:2")
        raise_http_exception: If True, raise HTTPException; else raise GPUConfigError
    
    Returns:
        True if valid
    
    Raises:
        HTTPException or GPUConfigError depending on raise_http_exception
    """
    try:
        if not gpu_config or ":" not in gpu_config:
            raise GPUConfigError("GPU config must be in format 'gpu_type:count'")
        
        gpu_type, count_str = gpu_config.rsplit(":", 1)
        
        if gpu_type not in VALID_GPU_TYPES:
            raise GPUConfigError(
                f"GPU type '{gpu_type}' not supported. "
                f"Valid types: {', '.join(VALID_GPU_TYPES)}"
            )
        
        count = int(count_str)
        if not 1 <= count <= 8:
            raise GPUConfigError("GPU count must be between 1 and 8")
        
        return True
    
    except GPUConfigError as e:
        if raise_http_exception:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(e))
        raise

@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    parameters_billions: float
    family: str
    context_length: int = 2048
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None

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


# Simple allocation rules based on model parameters
GPU_ALLOCATION_RULES = [
    (1.0, "L40S:1"),       # < 1B params
    (7.0, "L40S:1"),       # 1-7B params
    (13.0, "A100-80GB:1"), # 7-13B params
    (30.0, "A100-80GB:2"), # 13-30B params
    (70.0, "A100-80GB:4"), # 30-70B params
    (float('inf'), "A100-80GB:8"),  # > 70B params
]


def allocate_gpu_config(
    model_name: str,
    user_override: Optional[str] = None,
) -> str:
    """Allocate GPU configuration based on model size.
    
    Args:
        model_name: HuggingFace model name
        user_override: Optional user-specified GPU config
    
    Returns:
        GPU config string (e.g., "L40S:1")
    """
    if user_override:
        validate_gpu_config(user_override)
        logger.info(f"Using user-specified GPU config: {user_override}")
        return user_override
    
    model_info = get_model_info(model_name)
    if model_info is None:
        logger.warning(f"Unknown model {model_name}, using default GPU")
        return "L40S:1"
    
    params = model_info.parameters_billions
    
    for max_params, gpu_config in GPU_ALLOCATION_RULES:
        if params <= max_params:
            logger.info(
                f"Allocated {gpu_config} for {model_name} ({params}B params)"
            )
            return gpu_config
    
    return "A100-80GB:8"  # Fallback for very large models


def get_supported_models() -> Dict[str, ModelInfo]:
    """Get all supported models in the registry."""
    return MODEL_REGISTRY.copy()
