"""Custom loss functions for training."""
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


def causal_lm_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Standard causal language modeling loss."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["input_ids"],  # For causal LM, labels = input_ids
    )
    
    loss = outputs.loss
    
    if loss.dim() > 0:
        loss = loss.mean()
    
    metrics = {
        "loss": loss.item(),
        "perplexity": torch.exp(loss).item(),
    }
    
    return loss, metrics


def dpo_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
    reference_model: Optional[Any] = None,
    label_smoothing: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Direct Preference Optimization (DPO) loss.
    
    The batch should contain the following keys:
        - chosen_input_ids: Preferred completions
        - chosen_attention_mask: Mask for chosen
        - rejected_input_ids: Rejected completions  
        - rejected_attention_mask: Mask for rejected
    """
    # Get log probabilities for chosen completions
    chosen_outputs = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch.get("chosen_attention_mask"),
        labels=batch["chosen_input_ids"],
    )
    chosen_logps = -chosen_outputs.loss
    
    # Get log probabilities for rejected completions
    rejected_outputs = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch.get("rejected_attention_mask"),
        labels=batch["rejected_input_ids"],
    )
    rejected_logps = -rejected_outputs.loss
    
    # Get reference model log probabilities (if provided)
    if reference_model is not None:
        with torch.no_grad():
            ref_chosen_outputs = reference_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch.get("chosen_attention_mask"),
                labels=batch["chosen_input_ids"],
            )
            ref_chosen_logps = -ref_chosen_outputs.loss
            
            ref_rejected_outputs = reference_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch.get("rejected_attention_mask"),
                labels=batch["rejected_input_ids"],
            )
            ref_rejected_logps = -ref_rejected_outputs.loss
    else:
        # If no reference model, assume log probs are 0 (uniform prior)
        ref_chosen_logps = torch.zeros_like(chosen_logps)
        ref_rejected_logps = torch.zeros_like(rejected_logps)
    
    # Compute DPO loss
    # L_DPO = -log(σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x))))
    pi_logratios = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    
    if label_smoothing > 0:
        # Conservative DPO with label smoothing
        loss = -F.logsigmoid(beta * pi_logratios) * (1 - label_smoothing) - \
               F.logsigmoid(-beta * pi_logratios) * label_smoothing
    else:
        loss = -F.logsigmoid(beta * pi_logratios)
    
    loss = loss.mean()
    
    # Compute metrics
    with torch.no_grad():
        implicit_rewards = beta * pi_logratios
        chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
        
        # Preference accuracy: how often model prefers chosen over rejected
        accuracy = (chosen_logps > rejected_logps).float().mean()
    
    metrics = {
        "loss": loss.item(),
        "implicit_reward": implicit_rewards.mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "preference_accuracy": accuracy.item(),
    }
    
    return loss, metrics


# Loss function registry
LOSS_FUNCTIONS = {
    "causal_lm": causal_lm_loss,
    "dpo": dpo_loss,
}


def get_loss_function(loss_fn: str):
    """Get loss function by name."""
    if loss_fn not in LOSS_FUNCTIONS:
        available = ", ".join(LOSS_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown loss function: {loss_fn}. "
            f"Available: {available}"
        )
    return LOSS_FUNCTIONS[loss_fn]


def compute_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    loss_fn: str = "causal_lm",
    **loss_kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss using specified loss function."""
    loss_func = get_loss_function(loss_fn)
    return loss_func(model, batch, **loss_kwargs)
