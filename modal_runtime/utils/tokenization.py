"""Tokenization utilities using TRL data collators."""
import torch
from typing import Dict, Any, List, Optional


def get_data_collator(
    tokenizer: Any,
    loss_fn: str = "causal_lm",
    max_seq_length: int = 2048,
    **collator_kwargs
):
    """Get appropriate data collator for loss function.
    
    Args:
        tokenizer: HuggingFace tokenizer
        loss_fn: Loss function type (causal_lm, dpo, ppo, grpo)
        max_seq_length: Maximum sequence length
        **collator_kwargs: Additional collator arguments
        
    Returns:
        Data collator instance
        
    Examples:
        >>> # Causal LM
        >>> collator = get_data_collator(tokenizer, "causal_lm")
        
        >>> # DPO with completion-only masking
        >>> collator = get_data_collator(
        ...     tokenizer,
        ...     "dpo",
        ...     instruction_template="### Instruction:",
        ...     response_template="### Response:"
        ... )
    """
    if loss_fn == "causal_lm":
        from transformers import DataCollatorForLanguageModeling
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM (not masked LM)
        )
    
    elif loss_fn == "dpo":
        # DPO typically uses completion-only collator for masking prompts
        from trl import DataCollatorForCompletionOnlyLM
        
        instruction_template = collator_kwargs.get("instruction_template")
        response_template = collator_kwargs.get("response_template")
        
        if instruction_template and response_template:
            return DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                instruction_template=instruction_template,
                response_template=response_template,
            )
        else:
            # Fallback to standard collator
            from transformers import DataCollatorForLanguageModeling
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
    
    elif loss_fn in ["ppo", "grpo"]:
        # PPO and GRPO use standard padding collator
        from transformers import DataCollatorWithPadding
        return DataCollatorWithPadding(tokenizer=tokenizer)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def tokenize_batch(
    batch_data: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 2048,
    loss_fn: str = "causal_lm",
    **collator_kwargs
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch using appropriate data collator.
    
    For RL algorithms (DPO, PPO, GRPO), uses TRL's data collators.
    
    Args:
        batch_data: List of examples
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        loss_fn: Loss function type
        **collator_kwargs: Additional collator arguments
        
    Returns:
        Tokenized batch ready for training
        
    Examples:
        >>> # Standard causal LM
        >>> batch = [{"text": "Hello world"}]
        >>> tokens = tokenize_batch(batch, tokenizer)
        
        >>> # Chat format
        >>> batch = [{
        ...     "messages": [
        ...         {"role": "user", "content": "Hi"},
        ...         {"role": "assistant", "content": "Hello!"}
        ...     ]
        ... }]
        >>> tokens = tokenize_batch(batch, tokenizer)
    """
    # Get appropriate collator
    collator = get_data_collator(
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        max_seq_length=max_seq_length,
        **collator_kwargs
    )
    
    # Extract and format text from examples
    texts = []
    for example in batch_data:
        if "text" in example:
            texts.append(example["text"])
        elif "messages" in example:
            # Apply chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            else:
                raise ValueError(
                    "Tokenizer does not support chat templates. "
                    "Provide 'text' field instead of 'messages'."
                )
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
    
    # Note: For DPO/PPO/GRPO, the TRL trainers handle data collation internally
    # This function is primarily for standard causal LM training
    
    return encoded
