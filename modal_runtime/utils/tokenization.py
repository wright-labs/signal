"""Tokenization utilities for training."""
import torch
from typing import Dict, Any, List


def tokenize_batch(
    batch_data: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
    loss_fn: str = "causal_lm",
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of data."""
    # Handle preference pairs for DPO
    if loss_fn == "dpo":
        # Check if batch is preference pairs (has 'prompt', 'chosen', 'rejected')
        if batch_data and all('prompt' in ex and 'chosen' in ex and 'rejected' in ex 
                              for ex in batch_data):
            from modal_runtime.utils.preference_utils import format_preference_pairs_for_dpo
            return format_preference_pairs_for_dpo(
                preference_pairs=batch_data,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        
    # Standard tokenization for causal LM
    texts = []
    
    for example in batch_data:
        if "text" in example:
            texts.append(example["text"])
        elif "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        else:
            raise ValueError("Each example must have 'text' or 'messages' field")
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    return encoded

