# RL Algorithms & Training

Signal provides production-ready implementations of advanced RL algorithms for LLM fine-tuning, including PPO, DPO, GRPO, and more.

## Overview

Signal supports multiple RL algorithms optimized for language model training:

| Algorithm | Use Case | Key Features |
|-----------|----------|--------------|
| **Enhanced PPO** | General RL training | GAE, value function, entropy bonus, comprehensive metrics |
| **DPO** | Preference learning | Direct preference optimization, KL penalty |
| **GRPO** | Group-relative optimization | Per-group normalization, variance reduction |
| **Importance Sampling** | Off-policy learning | Replay buffer support, importance weight clipping |
| **Conservative PPO** | Safety-critical apps | Hard KL constraints, adaptive penalties |
| **Reward Modeling** | Preference model training | Bradley-Terry model, pairwise comparisons |

## Enhanced PPO

### Overview

Enhanced PPO is our flagship RL algorithm with:
- **Generalized Advantage Estimation (GAE)** for variance reduction
- **Value function** with optional clipping
- **Entropy bonus** for exploration
- **KL divergence** monitoring with reference model
- **Comprehensive metrics** for training visibility

### Usage

```python
result = await training_client.forward_backward(
    batch_data=ppo_batch,
    loss_fn="enhanced_ppo",
    loss_kwargs={
        "use_gae": True,  # Enable GAE
        "gamma": 0.99,  # Discount factor
        "gae_lambda": 0.95,  # GAE lambda parameter
        "clip_epsilon": 0.2,  # PPO clip parameter
        "value_loss_coef": 0.5,  # Value function weight
        "entropy_coef": 0.01,  # Entropy bonus weight
        "beta": 0.01,  # KL penalty coefficient
        "clip_value_loss": True,  # Clip value loss
        "reference_model": "meta-llama/Llama-3.2-1B",  # For KL penalty
    }
)
```

### Batch Format

```python
{
    "input_ids": [batch, seq_len],
    "attention_mask": [batch, seq_len],
    "old_log_probs": [batch, seq_len],  # From policy rollout
    "rewards": [batch, seq_len],  # Per-token rewards
    "values": [batch, seq_len],  # Value estimates (optional, auto-computed)
    "old_values": [batch, seq_len],  # For value clipping (optional)
}
```

### Returned Metrics

- `policy_loss`: Clipped surrogate objective loss
- `value_loss`: Value function MSE loss
- `entropy`: Policy entropy (higher = more exploratory)
- `kl_divergence`: KL from reference model
- `approx_kl`: Approximate KL (monitoring)
- `clip_fraction`: How often clipping is active (0-1)
- `advantage_mean`: Mean advantage
- `advantage_std`: Advantage standard deviation
- `explained_variance`: Value function quality (-1 to 1)
- `value_mean`, `return_mean`, `reward_mean`

### Generalized Advantage Estimation (GAE)

GAE provides a bias-variance tradeoff for advantage estimation:

**Formula:**
```
A^GAE(s,a) = Σ (γλ)^t δ_t
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Parameters:**
- `gamma` (0.99): Discount factor (higher = more future-focused)
- `gae_lambda` (0.95): GAE λ (higher = more variance, less bias)

**Tuning Guide:**
- λ=0: Only 1-step TD (low variance, high bias)
- λ=1: Monte Carlo (high variance, low bias)
- λ=0.95: Recommended default (good balance)

## Direct Preference Optimization (DPO)

### Overview

DPO trains models directly on preference data without reward modeling:

**Objective:**
```
L_DPO = -log σ(β * [log π_θ(y_w|x) - log π_ref(y_w|x) 
                   - log π_θ(y_l|x) + log π_ref(y_l|x)])
```

### Usage

```python
result = await training_client.forward_backward(
    batch_data=dpo_batch,
    loss_fn="dpo",
    loss_kwargs={
        "beta": 0.1,  # KL penalty coefficient
        "reference_model": "meta-llama/Llama-3.2-1B",
        "label_smoothing": 0.0,  # Conservative DPO (optional)
    }
)
```

### Batch Format

```python
{
    "chosen_input_ids": [batch, seq_len],
    "chosen_attention_mask": [batch, seq_len],
    "rejected_input_ids": [batch, seq_len],
    "rejected_attention_mask": [batch, seq_len],
}
```

### Returned Metrics

- `implicit_reward`: Implicit reward difference
- `chosen_reward`, `rejected_reward`: Individual rewards
- `reward_margin`: Difference between chosen/rejected
- `preference_accuracy`: How often model prefers chosen

## Group Relative Policy Optimization (GRPO)

### Overview

GRPO normalizes advantages per-group for better training stability with multi-sample rewards.

### Usage

```python
result = await training_client.forward_backward(
    batch_data=grpo_batch,
    loss_fn="grpo",
    loss_kwargs={
        "beta": 0.01,  # KL penalty
        "clip_epsilon": 0.2,  # PPO clipping
        "reference_model": "meta-llama/Llama-3.2-1B",
    }
)
```

### Batch Format

```python
{
    "prompt_ids": [batch_size, prompt_len],
    "response_ids": [batch_size, num_samples, response_len],
    "rewards": [batch_size, num_samples],
    "response_masks": [batch_size, num_samples, response_len],
}
```

## Importance Sampling

### Overview

Importance sampling enables off-policy learning from replay buffers:

**Formula:**
```
w = π_θ(a|s) / π_b(a|s)  # Importance weight
L = -E[w * A(s,a)]        # Weighted policy gradient
```

### Usage

```python
result = await training_client.forward_backward(
    batch_data=is_batch,
    loss_fn="importance_sampling",
    loss_kwargs={
        "beta": 0.1,  # KL penalty
        "clip_ratio": 10.0,  # Clip importance weights
        "reference_model": "meta-llama/Llama-3.2-1B",
    }
)
```

### Batch Format

```python
{
    "input_ids": [batch, seq_len],
    "attention_mask": [batch, seq_len],
    "behavior_log_probs": [batch],  # From behavior policy
    "rewards": [batch] or [batch, seq_len],
    "advantages": [batch] (optional),
}
```

### Returned Metrics

- `is_ratio_mean`, `is_ratio_max`, `is_ratio_min`: Importance weights
- `clipped_fraction`: How often weights are clipped

## Conservative PPO

### Overview

Conservative PPO enforces hard KL constraints to prevent policy collapse:

**Penalty:**
```
penalty = coef * max(0, KL - threshold)^2
```

### Usage

```python
result = await training_client.forward_backward(
    batch_data=cppo_batch,
    loss_fn="conservative_ppo",
    loss_kwargs={
        "conservative_coef": 0.1,  # Penalty coefficient
        "max_kl_divergence": 0.01,  # KL threshold
        "clip_epsilon": 0.2,
        "reference_model": "meta-llama/Llama-3.2-1B",
    }
)
```

### Returned Metrics

- `kl_violation`: Whether KL exceeded threshold
- `kl_penalty`: Applied penalty value
- `max_kl_threshold`: Current threshold

## Reward Modeling

### Overview

Train a reward model using Bradley-Terry preference model:

**Formula:**
```
P(y_w > y_l) = σ((r_w - r_l) / temperature)
```

### Usage

```python
result = await training_client.forward_backward(
    batch_data=rm_batch,
    loss_fn="reward_modeling",
    loss_kwargs={
        "margin": 1.0,  # Margin for preference
        "temperature": 0.1,  # Scaling temperature
    }
)
```

### Batch Format

```python
{
    "chosen_input_ids": [batch, seq_len],
    "chosen_attention_mask": [batch, seq_len],
    "rejected_input_ids": [batch, seq_len],
    "rejected_attention_mask": [batch, seq_len],
}
```

### Returned Metrics

- `preference_accuracy`: How often chosen > rejected
- `reward_margin`: Mean reward difference
- `chosen_reward_mean`, `rejected_reward_mean`
- `chosen_reward_std`, `rejected_reward_std`

## Complete RLHF Pipeline

### Three-Stage RLHF

```python
# Stage 1: Supervised Fine-Tuning
for batch in sft_data:
    await training_client.forward_backward(batch, "causal_lm")
    await training_client.optim_step()

# Stage 2: Reward Model Training
reward_client = client.get_training_client(reward_run_id)
for batch in preference_data:
    await reward_client.forward_backward(batch, "reward_modeling")
    await reward_client.optim_step()

# Stage 3: PPO Policy Training
for batch in rl_data:
    await training_client.forward_backward(batch, "enhanced_ppo", {
        "use_gae": True,
        "reference_model": "meta-llama/Llama-3.2-1B",
    })
    await training_client.optim_step()
```

## Hyperparameter Tuning

### PPO Hyperparameters

| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| `gamma` | 0.95-0.99 | 0.99 | Discount factor |
| `gae_lambda` | 0.9-0.98 | 0.95 | GAE λ |
| `clip_epsilon` | 0.1-0.3 | 0.2 | PPO clip |
| `value_loss_coef` | 0.5-1.0 | 0.5 | Value loss weight |
| `entropy_coef` | 0.001-0.01 | 0.01 | Entropy bonus |
| `beta` | 0.001-0.1 | 0.01 | KL penalty |
| `learning_rate` | 1e-6 to 1e-4 | 1e-5 | Lower for RL stability |

### DPO Hyperparameters

| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| `beta` | 0.05-0.5 | 0.1 | Controls reward scale |
| `label_smoothing` | 0.0-0.2 | 0.0 | Conservative DPO |

### Stability Monitoring

Watch these metrics for training stability:

- **KL divergence** < 0.05 (if too high, reduce learning rate or increase beta)
- **Clip fraction** = 0.1-0.3 (if too low, reduce clip_epsilon)
- **Explained variance** > 0.5 (if too low, improve value function)
- **Advantage std** ~1.0 (normalized by default)

## See Also

- [Futures Architecture](Futures-Architecture.md)
- [Reference Models](Reference-Models.md)
- [Metrics & Monitoring](Metrics.md)
- [Example: PPO with GAE](../client/examples/ppo_with_gae.py)

