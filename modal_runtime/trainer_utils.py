"""Training utilities for model training."""
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
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    import bitsandbytes as bnb
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == "adamw_8bit":
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
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
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
    
    return optimizer


def tokenize_batch(
    batch_data: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
    loss_fn: str = "causal_lm",
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of data.
    
    Args:
        batch_data: List of examples with 'text', 'messages', or preference pairs
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        loss_fn: Loss function being used (affects tokenization)
        
    Returns:
        Tokenized batch
    """
    # Handle preference pairs for DPO/reward modeling
    if loss_fn in ["dpo", "reward_modeling"]:
        # Check if batch is preference pairs (has 'prompt', 'chosen', 'rejected')
        if batch_data and all('prompt' in ex and 'chosen' in ex and 'rejected' in ex 
                              for ex in batch_data):
            from modal_runtime.preference_utils import format_preference_pairs_for_dpo
            return format_preference_pairs_for_dpo(
                preference_pairs=batch_data,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
    
    # Standard tokenization for causal LM, PPO, etc.
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
    
    Args:
        model: Model instance
        batch: Tokenized batch
        accumulate: Whether to accumulate gradients
        loss_fn: Loss function to use (causal_lm, dpo, reward_modeling, ppo)
        loss_kwargs: Additional arguments for loss function
        
    Returns:
        Tuple of (loss, metrics_dict including gradient_stats)
    """
    from modal_runtime.loss_functions import compute_loss
    
    if loss_kwargs is None:
        loss_kwargs = {}
    
    # Move batch to device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Compute loss using specified loss function
    loss, loss_metrics = compute_loss(model, batch, loss_fn=loss_fn, **loss_kwargs)
    
    # Backward pass
    loss.backward()
    
    # Compute gradient statistics
    grad_norms = []
    grad_values = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
            grad_values.extend(param.grad.flatten().tolist())
    
    grad_stats = {
        "grad_norm": sum(grad_norms) if grad_norms else 0.0,
        "grad_max": max(grad_values) if grad_values else 0.0,
        "grad_min": min(grad_values) if grad_values else 0.0,
        "grad_mean": sum(grad_values) / len(grad_values) if grad_values else 0.0,
    }
    
    # Merge loss metrics and gradient stats
    all_metrics = {**loss_metrics, **grad_stats}
    
    return loss.item(), all_metrics


def save_gradients(
    model: Any,
    save_path: str,
):
    """Save model gradients to disk.
    
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
    
    torch.save(gradients, save_path)


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
    
    for name, param in model.named_parameters():
        if name in gradients:
            if param.grad is None:
                param.grad = gradients[name].to(param.device)
            else:
                param.grad.add_(gradients[name].to(param.device))


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
        base_model_name: Base model name
        save_path: Path to save merged model
        tokenizer: Optional tokenizer to save
        framework: Framework used
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if framework == "unsloth":
        from unsloth import FastLanguageModel
        # Unsloth has special merge method
        model.save_pretrained_merged(
            str(save_path),
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        # For PEFT, merge and save
        from peft import PeftModel
        
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(save_path)
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)


def get_run_paths(user_id: str, run_id: str, base_path: str = "/data") -> Dict[str, Path]:
    """Get standard paths for a training run.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        base_path: Base storage path
        
    Returns:
        Dictionary of paths
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
        return json.load(f)


def find_latest_checkpoint(lora_adapters_path: Path, target_step: Optional[int] = None) -> Optional[Path]:
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
    
    # Find the most recent checkpoint
    checkpoints = sorted(lora_adapters_path.glob("step_*"))
    if checkpoints:
        return checkpoints[-1]
    
    return None

