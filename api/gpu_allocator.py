"""Automatic GPU allocation based on model size."""
import re
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


def estimate_model_parameters(model_name: str) -> Optional[int]:
    """Extract parameter count from model name or query HuggingFace API.
    
    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B")
        
    Returns:
        Parameter count in billions, or None if cannot be determined
    """
    # Strategy 1: Parse from model name
    # Common patterns: "7B", "8B", "70B", "1.5B", etc.
    patterns = [
        r'(\d+\.?\d*)B',  # Matches "8B", "1.5B", "70B"
        r'(\d+\.?\d*)b',  # Lowercase variant
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            try:
                param_count = float(match.group(1))
                logger.info(f"Extracted {param_count}B parameters from model name: {model_name}")
                return param_count
            except ValueError:
                continue
    
    # Strategy 2: Query HuggingFace API
    try:
        api_url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            
            # Try to find parameter count in model card metadata
            if 'config' in model_info and model_info['config']:
                config = model_info['config']
                # Some models have "num_parameters" or similar fields
                for key in ['num_parameters', 'n_params', 'parameters']:
                    if key in config:
                        # Convert to billions
                        param_count = config[key] / 1_000_000_000
                        logger.info(f"Got {param_count}B parameters from HF API for {model_name}")
                        return param_count
            
            # Try tags (some models have parameter count in tags)
            if 'tags' in model_info:
                for tag in model_info['tags']:
                    for pattern in patterns:
                        match = re.search(pattern, tag)
                        if match:
                            param_count = float(match.group(1))
                            logger.info(f"Got {param_count}B parameters from HF tags for {model_name}")
                            return param_count
    except Exception as e:
        logger.warning(f"Failed to query HuggingFace API for {model_name}: {e}")
    
    # Strategy 3: Fallback - assume large model for safety
    logger.warning(f"Could not determine parameter count for {model_name}, using conservative estimate")
    return None


def allocate_gpu_config(
    model_name: str,
    user_override: Optional[str] = None,
    parameter_count: Optional[float] = None
) -> str:
    """Allocate GPU configuration based on model size.
    
    Args:
        model_name: HuggingFace model ID
        user_override: User-specified GPU config (takes precedence)
        parameter_count: Optional pre-computed parameter count in billions
        
    Returns:
        GPU config string (e.g., "L40S:1", "A100-80GB:4")
        
    Allocation rules:
        - < 1B params: L40S:1
        - 1B - 3B: L40S:1
        - 3B - 7B: L40S:1
        - 7B - 13B: A100-80GB:1
        - 13B - 30B: A100-80GB:2
        - 30B - 70B: A100-80GB:4
        - > 70B: A100-80GB:8
    """
    # User override takes precedence
    if user_override:
        logger.info(f"Using user-specified GPU config: {user_override}")
        return user_override
    
    # Estimate parameter count if not provided
    if parameter_count is None:
        parameter_count = estimate_model_parameters(model_name)
    
    # Allocate based on parameter count
    if parameter_count is None:
        # Conservative default for unknown models
        gpu_config = "A100-80GB:2"
        logger.warning(
            f"Unknown model size for {model_name}, using conservative allocation: {gpu_config}"
        )
        return gpu_config
    
    # Allocation logic
    if parameter_count < 1.0:
        gpu_config = "L40S:1"
    elif parameter_count < 3.0:
        gpu_config = "L40S:1"
    elif parameter_count < 7.0:
        gpu_config = "L40S:1"
    elif parameter_count < 13.0:
        gpu_config = "A100-80GB:1"
    elif parameter_count < 30.0:
        gpu_config = "A100-80GB:2"
    elif parameter_count < 70.0:
        gpu_config = "A100-80GB:4"
    else:
        gpu_config = "A100-80GB:8"
    
    logger.info(
        f"Auto-allocated GPU config for {model_name} ({parameter_count}B params): {gpu_config}"
    )
    return gpu_config


def validate_gpu_config(gpu_config: str) -> bool:
    """Validate GPU configuration format.
    
    Args:
        gpu_config: GPU config string (e.g., "L40S:1", "A100-80GB:4")
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If invalid format
    """
    valid_gpus = ["L40S", "A100", "A100-80GB", "H100", "T4", "A10G"]
    
    if ':' not in gpu_config:
        raise ValueError("GPU config must be in format 'gpu_type:count' (e.g., 'L40S:1')")
    
    gpu_type, count_str = gpu_config.rsplit(':', 1)
    
    if gpu_type not in valid_gpus:
        raise ValueError(
            f"GPU type '{gpu_type}' not supported. Valid types: {', '.join(valid_gpus)}"
        )
    
    try:
        count = int(count_str)
    except ValueError:
        raise ValueError("GPU count must be a valid integer")
    
    if count < 1 or count > 8:
        raise ValueError("GPU count must be between 1 and 8")
    
    return True

