"""Utilities for preference-based training (DPO, RLHF, etc)."""
import torch
from typing import Dict, Any, List


def format_preference_pairs_for_dpo(
    preference_pairs: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Format preference pairs for DPO training.
    {
        "chosen_input_ids": [batch_size, seq_len],
        "chosen_attention_mask": [batch_size, seq_len],
        "rejected_input_ids": [batch_size, seq_len],
        "rejected_attention_mask": [batch_size, seq_len],
    }
    """
    chosen_texts = []
    rejected_texts = []
    
    for pair in preference_pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_text = f"{prompt}{chosen}"
        rejected_text = f"{prompt}{rejected}"
        
        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)
    
    chosen_encoded = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    rejected_encoded = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
    }


def format_preference_pairs_with_chat_template(
    preference_pairs: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Format preference pairs using chat template for DPO training.
    """
    if not hasattr(tokenizer, 'apply_chat_template'):
        return format_preference_pairs_for_dpo(
            preference_pairs, tokenizer, max_seq_length
        )
    
    chosen_texts = []
    rejected_texts = []
    
    for pair in preference_pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        messages_chosen = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        messages_rejected = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]
        
        chosen_text = tokenizer.apply_chat_template(
            messages_chosen,
            tokenize=False,
            add_generation_prompt=False,
        )
        rejected_text = tokenizer.apply_chat_template(
            messages_rejected,
            tokenize=False,
            add_generation_prompt=False,
        )

        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)

    chosen_encoded = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    rejected_encoded = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
    }


def format_grpo_samples(
    grpo_data: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Format GRPO samples for training.

    Args:
        grpo_data: List of dicts with 'prompt', 'responses', 'rewards' keys
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length

    Returns:
        Dictionary with:
        {
            "prompt_ids": [batch_size, prompt_len],
            "response_ids": [batch_size, num_samples, response_len],
            "rewards": [batch_size, num_samples],
            "response_masks": [batch_size, num_samples, response_len],
        }
    """
    prompt_texts = []
    all_responses = []
    all_rewards = []

    for example in grpo_data:
        prompt = example["prompt"]
        responses = example["responses"]  # List of response strings
        rewards = example["rewards"]      # List of reward values

        prompt_texts.append(prompt)
        all_responses.append(responses)
        all_rewards.append(rewards)

    # Tokenize prompts
    prompt_encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # Tokenize all responses for each prompt
    batch_size = len(grpo_data)
    max_responses = max(len(responses) for responses in all_responses)

    # Pad response lists to same length
    padded_responses = []
    padded_rewards = []
    response_masks = []

    for responses, rewards in zip(all_responses, all_rewards):
        # Pad responses to max_responses length
        padded_resp = responses + [""] * (max_responses - len(responses))
        padded_rew = rewards + [0.0] * (max_responses - len(rewards))

        padded_responses.append(padded_resp)
        padded_rewards.append(padded_rew)

        # Create response masks (1 for actual responses, 0 for padding)
        mask = [1] * len(responses) + [0] * (max_responses - len(responses))
        response_masks.append(mask)

    # Tokenize all responses
    flat_responses = [resp for responses in padded_responses for resp in responses]
    response_encoded = tokenizer(
        flat_responses,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # Reshape to [batch_size, max_responses, seq_len]
    # response_encoded["input_ids"] has shape [batch_size * max_responses, seq_len]
    # We want [batch_size, max_responses, seq_len]
    response_ids = response_encoded["input_ids"].view(batch_size, max_responses, -1)
    response_attention = response_encoded["attention_mask"].view(batch_size, max_responses, -1)

    return {
        "prompt_ids": prompt_encoded["input_ids"],
        "response_ids": response_ids,
        "rewards": torch.tensor(padded_rewards),
        "response_masks": torch.tensor(response_masks),
    }


def format_ppo_samples(
    ppo_data: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    Format PPO samples for training.

    Args:
        ppo_data: List of dicts with 'prompt', 'response', 'reward', 'value' keys
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length

    Returns:
        Dictionary with:
        {
            "prompt_ids": [batch_size, prompt_len],
            "response_ids": [batch_size, response_len],
            "rewards": [batch_size],
            "response_masks": [batch_size, response_len],
            "values": [batch_size],
        }
    """
    prompt_texts = []
    response_texts = []
    rewards = []
    values = []

    for example in ppo_data:
        # Combine prompt + response for full sequence
        prompt = example["prompt"]
        response = example["response"]
        full_text = f"{prompt}{response}"

        prompt_texts.append(prompt)
        response_texts.append(full_text)
        rewards.append(example["reward"])
        values.append(example["value"])

    # Tokenize prompts and responses
    prompt_encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    response_encoded = tokenizer(
        response_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    return {
        "prompt_ids": prompt_encoded["input_ids"],
        "response_ids": response_encoded["input_ids"],
        "rewards": torch.tensor(rewards),
        "response_masks": response_encoded["attention_mask"],
        "values": torch.tensor(values),
    }

