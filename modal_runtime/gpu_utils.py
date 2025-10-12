"""GPU configuration utilities for Modal."""
from typing import Tuple


def parse_gpu_config(gpu_config: str) -> Tuple[str, int]:
    """Parse GPU configuration string into type and count."""
    if ":" in gpu_config:
        parts = gpu_config.split(":")
        gpu_type = parts[0]
        gpu_count = int(parts[1])
    else:
        gpu_type = gpu_config
        gpu_count = 1
    
    return gpu_type, gpu_count


def format_gpu_config(gpu_type: str, gpu_count: int) -> str:
    """Format GPU type and count into Modal-compatible string."""
    return f"{gpu_type}:{gpu_count}"