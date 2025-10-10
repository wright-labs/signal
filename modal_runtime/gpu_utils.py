"""GPU configuration utilities for Modal."""
from typing import Tuple


def parse_gpu_config(gpu_config: str) -> Tuple[str, int]:
    """Parse GPU configuration string into type and count.
    
    Args:
        gpu_config: GPU config string (e.g., "a100-80gb:4" or "l40s:1")
        
    Returns:
        Tuple of (gpu_type, gpu_count)
        
    Examples:
        >>> parse_gpu_config("a100-80gb:4")
        ('a100-80gb', 4)
        >>> parse_gpu_config("l40s:1")
        ('l40s', 1)
        >>> parse_gpu_config("h100")
        ('h100', 1)
    """
    if ":" in gpu_config:
        parts = gpu_config.split(":")
        gpu_type = parts[0]
        gpu_count = int(parts[1])
    else:
        gpu_type = gpu_config
        gpu_count = 1
    
    return gpu_type, gpu_count


def format_gpu_config(gpu_type: str, gpu_count: int) -> str:
    """Format GPU type and count into Modal-compatible string.
    
    Args:
        gpu_type: GPU type (e.g., "a100-80gb", "l40s", "h100")
        gpu_count: Number of GPUs
        
    Returns:
        GPU config string for Modal (e.g., "a100-80gb:4")
    """
    return f"{gpu_type}:{gpu_count}"

