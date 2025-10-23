"""Checkpoint management utilities.

Note: Most checkpoint operations are handled by PEFT's save_pretrained().
These are minimal helpers for compatibility.
"""

from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_lora_checkpoint(
    model: Any,
    save_path: str,
    tokenizer: Optional[Any] = None,
):
    """Save LoRA checkpoint using PEFT."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # PEFT's save_pretrained() handles LoRA adapters
    model.save_pretrained(save_path)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)

    logger.info(f"✓ LoRA checkpoint saved to {save_path}")


def save_merged_model(
    model: Any,
    base_model_name: str,
    save_path: str,
    tokenizer: Optional[Any] = None,
    framework: str = "transformers",
):
    """Save merged model (base + LoRA)."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Merging LoRA weights with base model...")

    # Merge LoRA weights into base model (PEFT handles this)
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(save_path)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)

    logger.info(f"✓ Merged model saved to {save_path}")


def find_latest_checkpoint(
    lora_adapters_path: Path,
    target_step: Optional[int] = None,
) -> Optional[Path]:
    """Find the latest or target checkpoint."""
    if target_step is not None:
        checkpoint_path = lora_adapters_path / f"step_{target_step}"
        if checkpoint_path.exists():
            return checkpoint_path
        return None

    # Find the most recent checkpoint
    checkpoints = sorted(
        lora_adapters_path.glob("step_*"),
        key=lambda p: int(p.name.split("_")[1])
        if p.name.split("_")[1].isdigit()
        else 0,
    )

    if checkpoints:
        return checkpoints[-1]

    return None
