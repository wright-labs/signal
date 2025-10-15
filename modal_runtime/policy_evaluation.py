"""Policy evaluation utilities for RL training.

This module provides comprehensive policy evaluation metrics including:
- KL divergence from reference policy
- Model perplexity
- Generation diversity
- Response quality metrics
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import math


def compute_kl_divergence(
    model: Any,
    reference_model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute KL divergence between model and reference model.
    
    KL(π_θ || π_ref) = E[log π_θ(a|s) - log π_ref(a|s)]
    
    Args:
        model: Current policy model
        reference_model: Reference policy model
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        
    Returns:
        KL divergence value
    """
    with torch.no_grad():
        # Get log probs from current model
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        log_probs = -outputs.loss
        
        # Get log probs from reference model
        ref_outputs = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        ref_log_probs = -ref_outputs.loss
        
        # Compute KL divergence
        kl_div = (log_probs - ref_log_probs).mean()
    
    return kl_div.item()


def compute_perplexity(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute model perplexity on given inputs.
    
    Perplexity = exp(cross_entropy_loss)
    
    Args:
        model: Model to evaluate
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        
    Returns:
        Perplexity value
    """
    with torch.no_grad():
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()


def compute_generation_diversity(
    generated_texts: List[str],
    tokenizer: Any = None,
) -> Dict[str, float]:
    """Compute diversity metrics for generated text.
    
    Metrics include:
    - Unique token ratio
    - Unique n-gram ratios (bigrams, trigrams)
    - Self-BLEU (lower is more diverse)
    
    Args:
        generated_texts: List of generated text strings
        tokenizer: Optional tokenizer for tokenization
        
    Returns:
        Dictionary of diversity metrics
    """
    if len(generated_texts) == 0:
        return {
            "unique_token_ratio": 0.0,
            "unique_bigram_ratio": 0.0,
            "unique_trigram_ratio": 0.0,
            "avg_length": 0.0,
        }
    
    # Tokenize texts
    if tokenizer is not None:
        tokenized = [tokenizer.encode(text) for text in generated_texts]
    else:
        # Simple whitespace tokenization
        tokenized = [text.split() for text in generated_texts]
    
    # Compute unique token ratio
    all_tokens = [token for tokens in tokenized for token in tokens]
    unique_tokens = set(all_tokens)
    unique_token_ratio = len(unique_tokens) / len(all_tokens) if all_tokens else 0.0
    
    # Compute unique bigram ratio
    all_bigrams = []
    for tokens in tokenized:
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        all_bigrams.extend(bigrams)
    unique_bigrams = set(all_bigrams)
    unique_bigram_ratio = len(unique_bigrams) / len(all_bigrams) if all_bigrams else 0.0
    
    # Compute unique trigram ratio
    all_trigrams = []
    for tokens in tokenized:
        trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
        all_trigrams.extend(trigrams)
    unique_trigrams = set(all_trigrams)
    unique_trigram_ratio = len(unique_trigrams) / len(all_trigrams) if all_trigrams else 0.0
    
    # Compute average length
    avg_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    
    return {
        "unique_token_ratio": unique_token_ratio,
        "unique_bigram_ratio": unique_bigram_ratio,
        "unique_trigram_ratio": unique_trigram_ratio,
        "avg_length": avg_length,
        "num_texts": len(generated_texts),
    }


def compute_entropy(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute policy entropy.
    
    Higher entropy = more exploratory policy.
    
    Args:
        model: Model to evaluate
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        
    Returns:
        Entropy value
    """
    with torch.no_grad():
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    
    return entropy.item()


def evaluate_policy(
    model: Any,
    tokenizer: Any,
    eval_prompts: List[str],
    reference_model: Optional[Any] = None,
    max_tokens: int = 100,
    num_samples_per_prompt: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, float]:
    """Comprehensive policy evaluation.
    
    Args:
        model: Policy model to evaluate
        tokenizer: Tokenizer
        eval_prompts: List of evaluation prompts
        reference_model: Optional reference model for KL divergence
        max_tokens: Maximum tokens to generate
        num_samples_per_prompt: Number of samples per prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        Dictionary with evaluation metrics:
        - kl_divergence: KL from reference (if reference_model provided)
        - perplexity: Model perplexity
        - entropy: Policy entropy
        - diversity metrics: unique token/bigram/trigram ratios
        - avg_length: Average generation length
    """
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    
    all_metrics = {}
    
    # Generate samples
    all_generated_texts = []
    all_input_ids = []
    
    for prompt in eval_prompts:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate multiple samples
        for _ in range(num_samples_per_prompt):
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            all_generated_texts.append(generated_text)
            all_input_ids.append(generated_ids)
    
    # Compute diversity metrics
    diversity_metrics = compute_generation_diversity(all_generated_texts, tokenizer)
    all_metrics.update(diversity_metrics)
    
    # Compute perplexity on generated samples
    if all_input_ids:
        # Stack all generated IDs
        max_len = max(ids.shape[1] for ids in all_input_ids)
        padded_ids = []
        for ids in all_input_ids:
            if ids.shape[1] < max_len:
                padding = torch.full(
                    (ids.shape[0], max_len - ids.shape[1]),
                    tokenizer.pad_token_id,
                    device=ids.device
                )
                ids = torch.cat([ids, padding], dim=1)
            padded_ids.append(ids)
        
        stacked_ids = torch.cat(padded_ids, dim=0)
        
        # Compute perplexity
        perplexity = compute_perplexity(model, stacked_ids)
        all_metrics["perplexity"] = perplexity
        
        # Compute entropy
        entropy = compute_entropy(model, stacked_ids)
        all_metrics["entropy"] = entropy
        
        # Compute KL divergence if reference model provided
        if reference_model is not None:
            kl_div = compute_kl_divergence(model, reference_model, stacked_ids)
            all_metrics["kl_divergence"] = kl_div
    
    return all_metrics


def compare_policies(
    model1: Any,
    model2: Any,
    tokenizer: Any,
    eval_prompts: List[str],
    max_tokens: int = 100,
) -> Dict[str, Any]:
    """Compare two policies on evaluation prompts.
    
    Args:
        model1: First policy model
        model2: Second policy model
        tokenizer: Tokenizer
        eval_prompts: List of evaluation prompts
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with comparison metrics
    """
    model1.eval()
    model2.eval()
    
    # Evaluate both models
    metrics1 = evaluate_policy(
        model1,
        tokenizer,
        eval_prompts,
        reference_model=model2,
        max_tokens=max_tokens,
    )
    
    metrics2 = evaluate_policy(
        model2,
        tokenizer,
        eval_prompts,
        reference_model=model1,
        max_tokens=max_tokens,
    )
    
    return {
        "model1": metrics1,
        "model2": metrics2,
        "kl_model1_to_model2": metrics1.get("kl_divergence", 0.0),
        "kl_model2_to_model1": metrics2.get("kl_divergence", 0.0),
        "perplexity_diff": metrics1.get("perplexity", 0.0) - metrics2.get("perplexity", 0.0),
    }


def compute_reward_statistics(
    rewards: List[float],
) -> Dict[str, float]:
    """Compute statistics on reward distribution.
    
    Args:
        rewards: List of reward values
        
    Returns:
        Dictionary with reward statistics
    """
    if not rewards:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }
    
    import numpy as np
    
    rewards_array = np.array(rewards)
    
    return {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "median": float(np.median(rewards_array)),
        "p25": float(np.percentile(rewards_array, 25)),
        "p75": float(np.percentile(rewards_array, 75)),
        "p90": float(np.percentile(rewards_array, 90)),
        "p99": float(np.percentile(rewards_array, 99)),
    }

