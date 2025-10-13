"""Optimizer utilities for training."""
import torch
from pathlib import Path
from typing import Any


def setup_optimizer(
    model: Any,
    optimizer_type: str = "adamw_8bit",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Setup optimizer for training."""
    # Get trainable parameters (only LoRA adapters should be trainable)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    print(f"Setting up {optimizer_type} optimizer...")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Trainable params: {len(trainable_params)}")
    
    if optimizer_type == "adamw_8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}. Use 'adamw' or 'adamw_8bit'")
    
    return optimizer


def save_optimizer_state(
    optimizer: torch.optim.Optimizer,
    save_path: str,
):
    """Save optimizer state to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(optimizer.state_dict(), save_path)
    print(f"✓ Optimizer state saved to {save_path}")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    load_path: str,
) -> torch.optim.Optimizer:
    """Load optimizer state from disk."""
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Optimizer state not found: {load_path}")
    
    state_dict = torch.load(load_path, map_location="cpu")
    optimizer.load_state_dict(state_dict)
    
    print(f"✓ Optimizer state loaded from {load_path}")
    return optimizer

