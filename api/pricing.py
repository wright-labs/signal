"""Cost calculation for Modal GPU usage and storage."""

from datetime import datetime
from typing import Optional
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_pricing_config():
    """Load pricing configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "pricing.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


_pricing_config = _load_pricing_config()
GPU_HOURLY_RATES = _pricing_config["gpu_hourly_rates"]
STORAGE_COST_PER_GB_MONTH = _pricing_config["storage_cost_per_gb_month"]


def parse_gpu_type(gpu_config: str) -> tuple[str, int]:
    """Parse GPU config string into type and count."""
    if ":" in gpu_config:
        gpu_type, count_str = gpu_config.split(":")
        return gpu_type, int(count_str)
    return gpu_config, 1


def get_gpu_hourly_rate(gpu_config: str) -> float:
    """Get hourly rate for GPU configuration."""
    gpu_type, count = parse_gpu_type(gpu_config)
    rate = GPU_HOURLY_RATES.get(gpu_type, 2.0)  # Default to $2/hr if unknown

    if gpu_type not in GPU_HOURLY_RATES:
        logger.warning(f"Unknown GPU type '{gpu_type}', using default rate ${rate}/hr")

    return rate * count


def calculate_storage_cost(storage_bytes: int, hours: float = 1.0) -> float:
    """Calculate storage cost for given bytes and duration.

    Args:
        storage_bytes: Storage size in bytes
        hours: Duration in hours

    Returns:
        Storage cost in USD
    """
    storage_gb = storage_bytes / (1024**3)
    storage_cost_per_hour = STORAGE_COST_PER_GB_MONTH / 730
    return storage_gb * storage_cost_per_hour * hours


def calculate_run_cost(
    gpu_config: str,
    started_at: datetime,
    ended_at: Optional[datetime] = None,
    storage_bytes: int = 0,
    include_storage: bool = True,
) -> dict:
    """Unified cost calculation function"""
    if ended_at is None:
        ended_at = datetime.now(started_at.tzinfo or None)

    # Calculate duration
    duration_seconds = (ended_at - started_at).total_seconds()
    duration_hours = max(0, duration_seconds / 3600.0)  # Never negative

    # Calculate GPU cost
    hourly_rate = get_gpu_hourly_rate(gpu_config)
    gpu_cost = hourly_rate * duration_hours

    # Calculate storage cost
    storage_cost = 0.0
    if include_storage and storage_bytes > 0:
        storage_cost = calculate_storage_cost(storage_bytes, duration_hours)

    total_cost = gpu_cost + storage_cost

    logger.debug(
        f"Cost calculation: {duration_hours:.3f}h @ ${hourly_rate:.2f}/h = ${gpu_cost:.4f}, "
        f"storage {storage_bytes / (1024**3):.2f}GB = ${storage_cost:.4f}, "
        f"total = ${total_cost:.4f}"
    )

    return {
        "gpu_cost": gpu_cost,
        "storage_cost": storage_cost,
        "total_cost": total_cost,
        "hours": duration_hours,
    }
