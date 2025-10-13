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

