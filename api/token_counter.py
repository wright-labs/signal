"""Accurate token counting using tiktoken for billing purposes.

This module provides utilities for counting tokens in text and chat messages
using OpenAI's tiktoken library. Accurate token counting is essential for:
- Proper billing of API usage
- Cost estimation
- Rate limiting based on tokens
"""
from typing import List, Dict
import logging

try:
    import tiktoken
except ImportError:
    tiktoken = None
    logging.warning("tiktoken not installed, falling back to word-based estimation")

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in a text string using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for tokenization (default: gpt-3.5-turbo)
               Uses cl100k_base encoding for unknown models
    
    Returns:
        Number of tokens in the text
        
    Examples:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("The quick brown fox", "gpt-4")
        4
    """
    if tiktoken is None:
        # Fallback to word-based estimation (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.3)
    
    try:
        # Try to get encoding for the specific model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        # This is used by gpt-4, gpt-3.5-turbo, and text-embedding-ada-002
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def count_tokens_messages(
    messages: List[Dict[str, str]], 
    model: str = "gpt-3.5-turbo"
) -> int:
    """Count tokens in chat messages format.
    
    Chat messages have additional formatting tokens depending on the model.
    This function accounts for the message structure overhead.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model name (affects token counting due to format differences)
    
    Returns:
        Total number of tokens including message formatting overhead
        
    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> count_tokens_messages(messages)
        20  # Approximate, includes formatting tokens
    
    Reference:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    if tiktoken is None:
        # Fallback estimation
        total_words = sum(len(msg.get("content", "").split()) for msg in messages)
        return int(total_words * 1.3) + len(messages) * 4  # Add overhead per message
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Token counting varies by model
    if model in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613"]:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # If there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model or "gpt-4" in model:
        # Use the most recent model's token counting
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        # Unknown model, use conservative estimate
        tokens_per_message = 3
        tokens_per_name = 1
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
    
    return num_tokens


def estimate_cost(
    prompt_tokens: int, 
    completion_tokens: int, 
    model: str = "gpt-3.5-turbo"
) -> float:
    """Estimate cost in USD based on token counts.
    
    This uses approximate pricing and should be updated as pricing changes.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model name for pricing lookup
    
    Returns:
        Estimated cost in USD
        
    Note:
        This is for estimation only. Actual pricing may vary.
        Update pricing table as needed.
    """
    # Approximate pricing (as of 2024, per 1M tokens)
    pricing = {
        "gpt-4": {"prompt": 30.00, "completion": 60.00},
        "gpt-4-32k": {"prompt": 60.00, "completion": 120.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        "gpt-3.5-turbo-16k": {"prompt": 3.00, "completion": 4.00},
    }
    
    # Find matching model (partial match for versioned models)
    model_pricing = None
    for key in pricing:
        if key in model:
            model_pricing = pricing[key]
            break
    
    if model_pricing is None:
        # Default to gpt-3.5-turbo pricing for unknown models
        model_pricing = pricing["gpt-3.5-turbo"]
        logger.warning(f"Unknown model '{model}', using gpt-3.5-turbo pricing")
    
    prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
    
    return prompt_cost + completion_cost

