"""Training utilities for model training.

This module provides utilities for:
- Setting up optimizers (AdamW, 8-bit AdamW)
- Tokenizing batches for training
- Computing forward/backward passes with various loss functions
- Saving/loading checkpoints, gradients, and optimizer states
"""
import os
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json


def setup_optimizer(
    model: Any,
    optimizer_type: str = "adamw_8bit",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Setup optimizer for training.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw' or 'adamw_8bit')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Optimizer instance
    """
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
    """Save optimizer state to disk.
    
    Args:
        optimizer: Optimizer instance
        save_path: Path to save optimizer state
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(optimizer.state_dict(), save_path)
    print(f"✓ Optimizer state saved to {save_path}")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    load_path: str,
) -> torch.optim.Optimizer:
    """Load optimizer state from disk.
    
    Args:
        optimizer: Optimizer instance
        load_path: Path to load optimizer state from
        
    Returns:
        Optimizer with loaded state
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Optimizer state not found: {load_path}")
    
    state_dict = torch.load(load_path, map_location="cpu")
    optimizer.load_state_dict(state_dict)
    
    print(f"✓ Optimizer state loaded from {load_path}")
    return optimizer


def tokenize_batch(
    batch_data: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
    loss_fn: str = "causal_lm",
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of data.
    
    Args:
        batch_data: List of examples with 'text' or 'messages' fields
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        loss_fn: Loss function being used (affects tokenization for DPO)
        
    Returns:
        Tokenized batch with input_ids, attention_mask, labels
    """
    # Handle preference pairs for DPO
    if loss_fn == "dpo":
        # Check if batch is preference pairs (has 'prompt', 'chosen', 'rejected')
        if batch_data and all('prompt' in ex and 'chosen' in ex and 'rejected' in ex 
                              for ex in batch_data):
            # Import DPO utilities if needed
            try:
                from modal_runtime.preference_utils import format_preference_pairs_for_dpo
                return format_preference_pairs_for_dpo(
                    preference_pairs=batch_data,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                )
            except ImportError:
                print("Warning: DPO utilities not available, falling back to standard tokenization")
    
    # Standard tokenization for causal LM
    texts = []
    
    for example in batch_data:
        if "text" in example:
            texts.append(example["text"])
        elif "messages" in example:
            # Apply chat template if messages format
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        else:
            raise ValueError("Each example must have 'text' or 'messages' field")
    
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    return encoded


def compute_forward_backward(
    model: Any,
    batch: Dict[str, torch.Tensor],
    accumulate: bool = False,
    loss_fn: str = "causal_lm",
    loss_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute forward and backward pass with custom loss function.
    
    This function:
    1. Moves batch to model device
    2. Computes loss using specified loss function
    3. Performs backward pass to compute gradients
    4. Collects gradient statistics
    
    Args:
        model: Model instance
        batch: Tokenized batch
        accumulate: Whether to accumulate gradients (don't zero_grad)
        loss_fn: Loss function to use (causal_lm, dpo, etc.)
        loss_kwargs: Additional arguments for loss function
        
    Returns:
        Tuple of (loss_value, metrics_dict with gradient_stats)
    """
    from modal_runtime.loss_functions import compute_loss
    
    if loss_kwargs is None:
        loss_kwargs = {}
    
    # Move batch to device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Verify we have trainable parameters
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found! Cannot compute gradients. "
            "This usually means LoRA adapters were not applied correctly."
        )
    
    # Compute loss using specified loss function
    loss, loss_metrics = compute_loss(model, batch, loss_fn=loss_fn, **loss_kwargs)
    
    # Verify loss requires grad
    if not loss.requires_grad:
        raise RuntimeError(
            f"Loss does not require grad! This means gradient flow is broken. "
            f"Loss tensor: {loss}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}. "
            f"This is likely due to missing prepare_model_for_kbit_training() for quantized models."
        )
    
    # Backward pass
    loss.backward()
    
    # Compute gradient statistics
    grad_norms = []
    grad_values = []
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            # Only collect sample values to avoid memory issues
            if len(grad_values) < 1000:
                grad_values.extend(param.grad.flatten().tolist()[:100])
    
    if params_with_grad == 0:
        raise RuntimeError(
            "Backward pass completed but no parameters have gradients! "
            "This should never happen after loss.backward(). Check model setup."
        )
    
    # Calculate gradient statistics
    total_grad_norm = sum(n**2 for n in grad_norms) ** 0.5  # L2 norm
    
    grad_stats = {
        "grad_norm": total_grad_norm,
        "grad_max": max(grad_values) if grad_values else 0.0,
        "grad_min": min(grad_values) if grad_values else 0.0,
        "grad_mean": sum(grad_values) / len(grad_values) if grad_values else 0.0,
        "params_with_grad": params_with_grad,
    }
    
    print(f"Forward-backward complete: loss={loss.item():.4f}, grad_norm={total_grad_norm:.4f}")
    
    # Merge loss metrics and gradient stats
    all_metrics = {**loss_metrics, **grad_stats}
    
    return loss.item(), all_metrics


def save_gradients(
    model: Any,
    save_path: str,
):
    """Save model gradients to disk.
    
    This is useful for the distributed training API where gradient computation
    and optimizer updates happen in separate function calls.
    
    Args:
        model: Model instance
        save_path: Path to save gradients
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.cpu()
    
    if len(gradients) == 0:
        raise RuntimeError("No gradients to save! Call backward() first.")
    
    torch.save(gradients, save_path)
    print(f"✓ Saved {len(gradients)} gradient tensors to {save_path}")


def load_gradients(
    model: Any,
    load_path: str,
):
    """Load gradients into model.
    
    Args:
        model: Model instance
        load_path: Path to load gradients from
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Gradients not found: {load_path}")
    
    gradients = torch.load(load_path, map_location="cpu")
    
    loaded_count = 0
    for name, param in model.named_parameters():
        if name in gradients:
            if param.grad is None:
                param.grad = gradients[name].to(param.device)
            else:
                param.grad.add_(gradients[name].to(param.device))
            loaded_count += 1
    
    print(f"✓ Loaded {loaded_count} gradient tensors from {load_path}")


def save_lora_checkpoint(
    model: Any,
    save_path: str,
    tokenizer: Optional[Any] = None,
):
    """Save LoRA checkpoint.
    
    Args:
        model: Model with LoRA adapters
        save_path: Path to save checkpoint
        tokenizer: Optional tokenizer to save
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(save_path)
    
    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    
    print(f"✓ LoRA checkpoint saved to {save_path}")


def save_merged_model(
    model: Any,
    base_model_name: str,
    save_path: str,
    tokenizer: Optional[Any] = None,
    framework: str = "transformers",
):
    """Save merged model (base + LoRA).
    
    Args:
        model: Model with LoRA adapters
        base_model_name: Base model name (for reference)
        save_path: Path to save merged model
        tokenizer: Optional tokenizer to save
        framework: Framework used (currently only 'transformers' supported)
    """
    from peft import PeftModel
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Merging LoRA weights with base model...")
    
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(save_path)
    
    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    
    print(f"✓ Merged model saved to {save_path}")


def get_run_paths(user_id: str, run_id: str, base_path: str = "/data") -> Dict[str, Path]:
    """Get standard paths for a training run.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        base_path: Base storage path (default: /data on Modal)
        
    Returns:
        Dictionary of paths for various run artifacts
    """
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
    """Save run configuration.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        config: Configuration dictionary
        base_path: Base storage path
    """
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
    """Load run configuration.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        base_path: Base storage path
        
    Returns:
        Configuration dictionary
    """
    paths = get_run_paths(user_id, run_id, base_path)
    
    if not paths["config"].exists():
        raise FileNotFoundError(f"Run config not found: {paths['config']}")
    
    with open(paths["config"], "r") as f:
        config = json.load(f)
    
    return config


def find_latest_checkpoint(
    lora_adapters_path: Path,
    target_step: Optional[int] = None,
) -> Optional[Path]:
    """Find the latest or target checkpoint.
    
    Args:
        lora_adapters_path: Path to LoRA adapters directory
        target_step: Optional specific step to find (if None, finds latest)
        
    Returns:
        Path to checkpoint, or None if not found
    """
    if target_step is not None:
        checkpoint_path = lora_adapters_path / f"step_{target_step}"
        if checkpoint_path.exists():
            return checkpoint_path
        return None
    
    # Find the most recent checkpoint
    checkpoints = sorted(
        lora_adapters_path.glob("step_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0
    )
    
    if checkpoints:
        return checkpoints[-1]
    
    return None
