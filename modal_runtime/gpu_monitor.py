"""Simple GPU monitoring using torch.cuda."""

import torch
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def get_gpu_stats() -> List[Dict[str, Any]]:
    """Get basic GPU memory stats."""
    if not torch.cuda.is_available():
        return []

    stats = []
    for i in range(torch.cuda.device_count()):
        stats.append(
            {
                "gpu_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory
                / 1024**3,
                "utilization_percent": (
                    torch.cuda.memory_allocated(i)
                    / torch.cuda.get_device_properties(i).total_memory
                    * 100
                ),
            }
        )
    return stats


def get_gpu_summary() -> Dict[str, Any]:
    """Get GPU summary across all GPUs."""
    stats = get_gpu_stats()
    if not stats:
        return {"num_gpus": 0}

    return {
        "num_gpus": len(stats),
        "total_memory_gb": sum(s["memory_total_gb"] for s in stats),
        "allocated_memory_gb": sum(s["memory_allocated_gb"] for s in stats),
        "reserved_memory_gb": sum(s["memory_reserved_gb"] for s in stats),
        "avg_utilization_percent": sum(s["utilization_percent"] for s in stats)
        / len(stats),
    }


def print_gpu_stats():
    """Print GPU stats in a readable format."""
    stats = get_gpu_stats()
    if not stats:
        logger.info("No GPUs available")
        return

    logger.info(f"GPU Stats ({len(stats)} GPU{'s' if len(stats) > 1 else ''})")

    for stat in stats:
        logger.info(f"GPU {stat['gpu_id']}: {stat['name']}")
        logger.info(
            f"  Memory: {stat['memory_allocated_gb']:.2f} GB / {stat['memory_total_gb']:.2f} GB"
        )
        logger.info(f"  Utilization: {stat['utilization_percent']:.1f}%")

    logger.info(f"{'=' * 60}\n")


# TODO: check this, not sure if this is the correct way to do it
# TODO: if i'm using accelerate, do I still need to do this?
# TODO: pretty sure it's completely unused
def setup_multi_gpu_model(model: Any, strategy: str = "data_parallel") -> Any:
    """Wrap model for multi-GPU using DataParallel."""
    if not torch.cuda.is_available():
        return model

    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        return model

    if strategy == "data_parallel":
        logger.info(f"Wrapping model with DataParallel ({num_gpus} GPUs)")
        return torch.nn.DataParallel(model)
    else:
        raise ValueError(
            f"Unknown multi-GPU strategy: {strategy}. "
            f"Only 'data_parallel' is supported."
        )
