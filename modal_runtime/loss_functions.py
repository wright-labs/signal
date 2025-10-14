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


def grpo_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.01,  # KL penalty coefficient
    clip_epsilon: float = 0.2,  # PPO clipping parameter
    reference_model: Optional[Any] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Group Relative Policy Optimization loss.

    The batch should contain:
        - prompt_ids: [batch_size, prompt_len]
        - response_ids: [batch_size, num_samples, response_len]
        - rewards: [batch_size, num_samples]
        - response_masks: [batch_size, num_samples, response_len]
    """
    batch_size, num_samples, seq_len = batch["response_ids"].shape

    # Flatten to process all responses at once
    flat_response_ids = batch["response_ids"].view(-1, seq_len)
    flat_response_masks = batch["response_masks"].view(-1, seq_len)

    # Get current policy log probabilities
    outputs = model(
        input_ids=flat_response_ids,
        attention_mask=flat_response_masks,
        labels=flat_response_ids,
    )
    log_probs = -outputs.loss
    log_probs = log_probs.view(batch_size, num_samples)

    # Get reference model log probabilities for KL penalty
    if reference_model is not None:
        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=flat_response_ids,
                attention_mask=flat_response_masks,
                labels=flat_response_ids,
            )
            ref_log_probs = -ref_outputs.loss
            ref_log_probs = ref_log_probs.view(batch_size, num_samples)
    else:
        ref_log_probs = torch.zeros_like(log_probs)

    # Compute group-relative advantages
    rewards = batch["rewards"]  # [batch_size, num_samples]

    # Per-group statistics (normalize by group)
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards - mean_rewards) / std_rewards

    # PPO-style clipped loss with group-relative advantages
    ratio = torch.exp(log_probs - ref_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL penalty
    kl_divergence = (ratio - 1 - torch.log(ratio)).mean()

    loss = policy_loss + beta * kl_divergence

    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_divergence": kl_divergence.item(),
        "mean_reward": rewards.mean().item(),
        "advantage_std": advantages.std().item(),
        "num_samples": num_samples,
    }

    return loss, metrics


def ppo_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    value_model: Optional[Any] = None,
    beta: float = 0.01,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,  # Discount factor for advantage estimation
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Proximal Policy Optimization loss with value function.

    The batch should contain:
        - prompt_ids: [batch_size, prompt_len]
        - response_ids: [batch_size, response_len]
        - rewards: [batch_size] (per-response rewards)
        - response_masks: [batch_size, response_len]
        - values: [batch_size] (value function outputs, if using critic)
    """
    # Current policy log probabilities
    outputs = model(
        input_ids=batch["response_ids"],
        attention_mask=batch["response_masks"],
        labels=batch["response_ids"],
    )
    log_probs = -outputs.loss  # [batch_size]

    # PPO clipped loss
    ratio = torch.exp(log_probs - batch.get("old_log_probs", log_probs))

    # Simple advantage calculation (reward - baseline)
    rewards = batch["rewards"]  # [batch_size]
    baseline = rewards.mean()  # Simple baseline = mean reward
    advantages = rewards - baseline

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (if using value model)
    value_loss = torch.tensor(0.0)
    if value_model is not None and "values" in batch:
        values = batch["values"]  # [batch_size]
        # Use rewards as target values (simplified)
        value_targets = rewards
        value_loss = F.mse_loss(values, value_targets)

    # Entropy bonus (encourage exploration)
    entropy = -(torch.exp(log_probs) * log_probs).mean()

    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "mean_advantage": advantages.mean().item(),
        "advantage_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
    }

    return loss, metrics


# Loss function registry
LOSS_FUNCTIONS = {
    "causal_lm": causal_lm_loss,
    "dpo": dpo_loss,
    "grpo": grpo_loss,
    "ppo": ppo_loss,
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
