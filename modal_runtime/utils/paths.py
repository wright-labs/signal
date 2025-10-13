"""Path utilities for volume organization."""
import json
from pathlib import Path
from typing import Dict, Any


def get_run_paths(user_id: str, run_id: str, base_path: str = "/data") -> Dict[str, Path]:
    """Get standard paths for a training run."""
    base = Path(base_path) / "runs" / user_id / run_id
    
    paths = {
        "base": base,
        "config": base / "config.json",
        "lora_adapters": base / "lora_adapters",
        "optimizer_state": base / "optimizer_state.pt",
        "gradients": base / "gradients",
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
    }
    
    return paths


def save_run_config(
    user_id: str,
    run_id: str,
    config: Dict[str, Any],
    base_path: str = "/data",
):
    """Save run configuration."""
    paths = get_run_paths(user_id, run_id, base_path)
    paths["base"].mkdir(parents=True, exist_ok=True)
    
    with open(paths["config"], "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Run config saved to {paths['config']}")


def load_run_config(
    user_id: str,
    run_id: str,
    base_path: str = "/data",
) -> Dict[str, Any]:
    """Load run configuration."""
    paths = get_run_paths(user_id, run_id, base_path)
    
    if not paths["config"].exists():
        raise FileNotFoundError(f"Run config not found: {paths['config']}")
    
    with open(paths["config"], "r") as f:
        config = json.load(f)
    
    return config

