"""Loss computation from model outputs (NOT from model itself).

This module computes losses from HuggingFace model outputs.
For RL algorithms (DPO, PPO, etc.), use TRL library instead.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any


def compute_loss_from_outputs(
    outputs,
    labels: torch.Tensor = None,
    loss_fn: str = "causal_lm",
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss from model outputs using HuggingFace's built-in losses.
    
    Args:
        outputs: Model outputs (from model(**inputs))
        labels: Ground truth labels (optional, may already be in outputs)
        loss_fn: Loss function type (causal_lm, dpo, ppo)
        **kwargs: Additional loss function arguments
        
    Returns:
        Tuple of (loss_tensor, metrics_dict)
        
    Examples:
        >>> outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        >>> loss, metrics = compute_loss_from_outputs(outputs, loss_fn="causal_lm")
    """
    if loss_fn == "causal_lm":
        # HuggingFace already computed loss if labels were provided
        if not hasattr(outputs, 'loss') or outputs.loss is None:
            raise ValueError(
                "Model outputs do not contain loss. "
                "Make sure to pass labels to the model."
            )
        
        loss = outputs.loss
        
        # Ensure loss is scalar
        if loss.dim() > 0:
            loss = loss.mean()
        
        metrics = {
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
        }
        
        return loss, metrics
    
    elif loss_fn == "dpo":
        # Use TRL's DPO implementation
        raise NotImplementedError(
            "DPO should use TRL's DPOTrainer. "
            "See: https://huggingface.co/docs/trl/dpo_trainer"
        )
    
    elif loss_fn == "ppo":
        # Use TRL's PPO implementation
        raise NotImplementedError(
            "PPO should use TRL's PPOTrainer. "
            "See: https://huggingface.co/docs/trl/ppo_trainer"
        )
    
    elif loss_fn == "grpo":
        # Use TRL's implementation
        raise NotImplementedError(
            "GRPO should use TRL's implementation. "
            "See: https://huggingface.co/docs/trl/"
        )
    
    elif loss_fn == "reward_modeling":
        # Use TRL's RewardTrainer
        raise NotImplementedError(
            "Reward modeling should use TRL's RewardTrainer. "
            "See: https://huggingface.co/docs/trl/reward_trainer"
        )
    
    else:
        raise ValueError(
            f"Unknown loss function: {loss_fn}. "
            f"Supported: causal_lm. "
            f"For RL losses, use TRL library."
        )


# Legacy compatibility - map old function names to new implementation
def compute_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    loss_fn: str = "causal_lm",
    **loss_kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Legacy compatibility wrapper.
    
    DEPRECATED: This function calls the model internally, which is confusing.
    Use compute_loss_from_outputs() instead and call the model explicitly.
    
    Args:
        model: The model (will be called internally)
        batch: Batch dictionary with input_ids, attention_mask, labels
        loss_fn: Loss function name
        **loss_kwargs: Additional loss arguments
        
    Returns:
        Tuple of (loss_tensor, metrics_dict)
    """
    import warnings
    warnings.warn(
        "compute_loss() is deprecated. "
        "Call model() explicitly and use compute_loss_from_outputs() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Call model
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels", batch["input_ids"]),
    )
    
    # Compute loss from outputs
    return compute_loss_from_outputs(outputs, batch.get("labels"), loss_fn, **loss_kwargs)
