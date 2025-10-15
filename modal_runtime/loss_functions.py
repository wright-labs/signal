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


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).
    
    GAE provides a trade-off between bias and variance in advantage estimation
    using exponentially-weighted average of n-step advantages.
    
    Args:
        rewards: [batch_size, seq_len] reward per timestep
        values: [batch_size, seq_len] value function estimates
        gamma: Discount factor (0.99 typical)
        lambda_: GAE lambda parameter (0.95 typical, higher = more variance)
    
    Returns:
        advantages: [batch_size, seq_len] GAE advantages
        returns: [batch_size, seq_len] discounted returns for value targets
    """
    batch_size, seq_len = rewards.shape
    
    # Compute TD residuals (delta_t)
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    values_next = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    deltas = rewards + gamma * values_next - values
    
    # Compute GAE using reverse iteration
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, device=rewards.device)
    
    for t in reversed(range(seq_len)):
        gae = deltas[:, t] + gamma * lambda_ * gae
        advantages[:, t] = gae
    
    # Compute returns: A_t = R_t - V_t => R_t = A_t + V_t
    returns = advantages + values
    
    return advantages, returns


def enhanced_ppo_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    reference_model: Optional[Any] = None,
    beta: float = 0.01,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_gae: bool = True,
    clip_value_loss: bool = True,
    max_grad_norm: Optional[float] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enhanced PPO with GAE, value function, entropy, and comprehensive metrics.
    
    Implements the full PPO algorithm with:
    - Clipped surrogate objective
    - Value function with optional clipping
    - GAE for advantage estimation
    - Entropy bonus for exploration
    - KL divergence monitoring
    - Comprehensive metrics
    
    The batch should contain:
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len]
        - old_log_probs: [batch, seq_len] (from policy rollout)
        - rewards: [batch, seq_len]
        - values: [batch, seq_len] (optional, computed if not provided)
        - old_values: [batch, seq_len] (for value clipping, optional)
    """
    # Get current policy log probabilities
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["input_ids"],
    )
    log_probs = -outputs.loss  # Negative loss = log prob
    
    # Expand to per-token if needed
    if log_probs.dim() == 0:
        log_probs = log_probs.unsqueeze(0).expand(batch["input_ids"].shape[0])
    
    # Get old log probs
    old_log_probs = batch.get("old_log_probs", log_probs.detach())
    if old_log_probs.dim() == 1:
        old_log_probs = old_log_probs.unsqueeze(1)
    
    # Compute value function estimates if not provided
    if "values" not in batch:
        # Use last hidden state mean as value estimate (simple baseline)
        with torch.no_grad():
            values = outputs.logits.mean(dim=-1).mean(dim=-1, keepdim=True)
            values = values.expand(-1, batch["input_ids"].shape[1])
    else:
        values = batch["values"]
    
    # Get rewards
    rewards = batch["rewards"]
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1).expand(-1, batch["input_ids"].shape[1])
    
    # Compute advantages using GAE
    if use_gae:
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values.detach(),
            gamma=gamma,
            lambda_=gae_lambda,
        )
    else:
        # Simple advantage: reward - value baseline
        returns = rewards  # Simplified
        advantages = rewards - values.detach()
    
    # Normalize advantages (per-batch)
    advantages_mean = advantages.mean()
    advantages_std = advantages.std() + 1e-8
    advantages_normalized = (advantages - advantages_mean) / advantages_std
    
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs.detach())
    
    # Clipped surrogate objective
    surr1 = ratio * advantages_normalized
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_normalized
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    if clip_value_loss and "old_values" in batch:
        # Clipped value loss
        old_values = batch["old_values"]
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -clip_epsilon,
            clip_epsilon
        )
        value_loss_unclipped = F.mse_loss(values, returns.detach())
        value_loss_clipped = F.mse_loss(value_pred_clipped, returns.detach())
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    else:
        value_loss = F.mse_loss(values, returns.detach())
    
    # Entropy bonus (encourage exploration)
    # Approximate entropy from log probs
    entropy = -(torch.exp(log_probs) * log_probs).mean()
    
    # KL divergence with reference model (if provided)
    kl_divergence = torch.tensor(0.0, device=log_probs.device)
    if reference_model is not None:
        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["input_ids"],
            )
            ref_log_probs = -ref_outputs.loss
            if ref_log_probs.dim() == 0:
                ref_log_probs = ref_log_probs.unsqueeze(0).expand(batch["input_ids"].shape[0])
        
        # KL(π_θ || π_ref) = E[log π_θ - log π_ref]
        kl_divergence = (log_probs - ref_log_probs).mean()
    
    # Total loss
    loss = (
        policy_loss
        + value_loss_coef * value_loss
        - entropy_coef * entropy
        + beta * kl_divergence
    )
    
    # Compute comprehensive metrics
    with torch.no_grad():
        # Clip fraction (how often clipping is active)
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
        
        # Approximate KL divergence (for monitoring)
        approx_kl = (log_probs - old_log_probs).mean()
        
        # Explained variance (value function quality)
        # EV = 1 - Var(returns - values) / Var(returns)
        y_pred = values.detach()
        y_true = returns.detach()
        var_y = y_true.var()
        explained_variance = 1 - (y_true - y_pred).var() / (var_y + 1e-8)
        explained_variance = torch.clamp(explained_variance, -1.0, 1.0)
    
    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "kl_divergence": kl_divergence.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "advantage_mean": advantages_mean.item(),
        "advantage_std": advantages_std.item(),
        "explained_variance": explained_variance.item(),
        "value_mean": values.mean().item(),
        "return_mean": returns.mean().item(),
        "reward_mean": rewards.mean().item(),
    }
    
    return loss, metrics


def importance_sampling_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    reference_model: Optional[Any] = None,
    beta: float = 0.1,
    clip_ratio: float = 10.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Importance sampling loss for off-policy RL.
    
    Enables learning from a replay buffer or offline data by correcting
    for the distribution mismatch using importance weights.
    
    The batch should contain:
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len]
        - behavior_log_probs: [batch] (log probs from behavior policy)
        - rewards: [batch] or [batch, seq_len]
        - advantages: [batch] or [batch, seq_len] (optional)
    """
    # Get current policy log probabilities
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["input_ids"],
    )
    current_log_probs = -outputs.loss
    
    if current_log_probs.dim() == 0:
        current_log_probs = current_log_probs.unsqueeze(0).expand(batch["input_ids"].shape[0])
    
    # Get behavior policy log probs
    behavior_log_probs = batch["behavior_log_probs"]
    if behavior_log_probs.dim() == 1 and current_log_probs.dim() == 2:
        behavior_log_probs = behavior_log_probs.unsqueeze(1)
    
    # Compute importance weights: w = π_θ(a|s) / π_b(a|s)
    log_importance_weights = current_log_probs - behavior_log_probs.detach()
    importance_weights = torch.exp(log_importance_weights)
    
    # Clip importance weights for stability
    importance_weights_clipped = torch.clamp(importance_weights, 1.0 / clip_ratio, clip_ratio)
    
    # Get advantages (or use rewards directly)
    if "advantages" in batch:
        advantages = batch["advantages"]
    else:
        advantages = batch["rewards"]
    
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    
    # Importance-weighted policy gradient
    policy_loss = -(importance_weights_clipped * advantages.detach()).mean()
    
    # KL penalty with reference model (optional)
    kl_penalty = torch.tensor(0.0, device=current_log_probs.device)
    if reference_model is not None:
        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["input_ids"],
            )
            ref_log_probs = -ref_outputs.loss
            if ref_log_probs.dim() == 0:
                ref_log_probs = ref_log_probs.unsqueeze(0).expand(batch["input_ids"].shape[0])
        
        kl_penalty = (current_log_probs - ref_log_probs).mean()
    
    loss = policy_loss + beta * kl_penalty
    
    # Metrics
    with torch.no_grad():
        clipped_fraction = (importance_weights.abs() > clip_ratio).float().mean()
    
    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_penalty": kl_penalty.item(),
        "is_ratio_mean": importance_weights.mean().item(),
        "is_ratio_max": importance_weights.max().item(),
        "is_ratio_min": importance_weights.min().item(),
        "clipped_fraction": clipped_fraction.item(),
        "advantage_mean": advantages.mean().item(),
    }
    
    return loss, metrics


def conservative_ppo_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    reference_model: Optional[Any] = None,
    conservative_coef: float = 0.1,
    max_kl_divergence: float = 0.01,
    clip_epsilon: float = 0.2,
    adaptive_kl: bool = True,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Conservative PPO with hard KL constraints.
    
    Prevents policy collapse by enforcing a maximum KL divergence from
    the reference policy. Useful for safety-critical applications.
    
    The batch should contain:
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len]
        - old_log_probs: [batch]
        - rewards: [batch]
        - advantages: [batch] (optional)
    """
    # Get current policy log probabilities
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["input_ids"],
    )
    log_probs = -outputs.loss
    
    if log_probs.dim() == 0:
        log_probs = log_probs.unsqueeze(0).expand(batch["input_ids"].shape[0])
    
    # Get old log probs
    old_log_probs = batch.get("old_log_probs", log_probs.detach())
    if old_log_probs.dim() == 1 and log_probs.dim() == 2:
        old_log_probs = old_log_probs.unsqueeze(1)
    
    # Get advantages
    if "advantages" in batch:
        advantages = batch["advantages"]
    else:
        rewards = batch["rewards"]
        baseline = rewards.mean()
        advantages = rewards - baseline
    
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs.detach())
    
    # Standard PPO clipped objective
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.detach()
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Compute KL divergence from reference
    kl_divergence = torch.tensor(0.0, device=log_probs.device)
    if reference_model is not None:
        with torch.no_grad():
            ref_outputs = reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["input_ids"],
            )
            ref_log_probs = -ref_outputs.loss
            if ref_log_probs.dim() == 0:
                ref_log_probs = ref_log_probs.unsqueeze(0).expand(batch["input_ids"].shape[0])
        
        kl_divergence = (log_probs - ref_log_probs).mean()
    
    # Conservative penalty (quadratic beyond threshold)
    kl_penalty = torch.tensor(0.0, device=log_probs.device)
    if kl_divergence > max_kl_divergence:
        kl_excess = kl_divergence - max_kl_divergence
        kl_penalty = conservative_coef * (kl_excess ** 2)
    
    loss = policy_loss + kl_penalty
    
    # Metrics
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
        kl_violation = (kl_divergence > max_kl_divergence).float()
    
    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_divergence": kl_divergence.item(),
        "kl_penalty": kl_penalty.item(),
        "kl_violation": kl_violation.item(),
        "clip_fraction": clip_fraction.item(),
        "max_kl_threshold": max_kl_divergence,
    }
    
    return loss, metrics


def reward_modeling_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    margin: float = 1.0,
    temperature: float = 0.1,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Bradley-Terry reward modeling loss for preference learning.
    
    Trains a reward model on pairwise preference data.
    The model learns to assign higher rewards to chosen completions
    than rejected completions.
    
    The batch should contain:
        - chosen_input_ids: [batch, seq_len]
        - chosen_attention_mask: [batch, seq_len]
        - rejected_input_ids: [batch, seq_len]
        - rejected_attention_mask: [batch, seq_len]
    """
    # Get reward scores for chosen completions
    chosen_outputs = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch.get("chosen_attention_mask"),
    )
    # Use last token logits mean as reward score
    chosen_rewards = chosen_outputs.logits[:, -1, :].mean(dim=-1)
    
    # Get reward scores for rejected completions
    rejected_outputs = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch.get("rejected_attention_mask"),
    )
    rejected_rewards = rejected_outputs.logits[:, -1, :].mean(dim=-1)
    
    # Bradley-Terry preference model loss
    # P(chosen > rejected) = sigmoid((r_chosen - r_rejected) / temperature)
    reward_diff = (chosen_rewards - rejected_rewards) / temperature
    
    # Maximize probability of choosing the chosen completion
    loss = -F.logsigmoid(reward_diff - margin).mean()
    
    # Metrics
    with torch.no_grad():
        # Preference accuracy: how often chosen > rejected
        preference_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        reward_margin = (chosen_rewards - rejected_rewards).mean()
    
    metrics = {
        "loss": loss.item(),
        "preference_accuracy": preference_accuracy.item(),
        "reward_margin": reward_margin.item(),
        "chosen_reward_mean": chosen_rewards.mean().item(),
        "rejected_reward_mean": rejected_rewards.mean().item(),
        "chosen_reward_std": chosen_rewards.std().item(),
        "rejected_reward_std": rejected_rewards.std().item(),
    }
    
    return loss, metrics


# Loss function registry
LOSS_FUNCTIONS = {
    "causal_lm": causal_lm_loss,
    "dpo": dpo_loss,
    "grpo": grpo_loss,
    "ppo": ppo_loss,
    "enhanced_ppo": enhanced_ppo_loss,
    "importance_sampling": importance_sampling_loss,
    "conservative_ppo": conservative_ppo_loss,
    "reward_modeling": reward_modeling_loss,
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
