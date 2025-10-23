"""Model loading utilities for transformers + PEFT.

This module provides utilities for loading models with LoRA adapters
using the Hugging Face transformers and PEFT libraries.

Key features:
- Supports quantized models (8-bit/4-bit) for memory efficiency
- Applies LoRA adapters for parameter-efficient fine-tuning
- Fixes gradient flow for quantized models with prepare_model_for_kbit_training()
"""

import os
import torch
from pathlib import Path
from typing import Tuple, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


# TODO: wait, I definitely want to kepe the base models on a permanent modal volume so I can quickly and easily load them no?
def load_model_and_tokenizer(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    max_seq_length: int = 2048,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    device_map: Optional[str] = "auto",
) -> Tuple[Any, Any]:
    """Load model and tokenizer for training.

    This function loads a causal language model for LoRA fine-tuning.

    By default uses full precision LoRA (bf16/fp16). Set load_in_4bit=True or
    load_in_8bit=True for QLoRA (quantized LoRA) to save memory."""
    # TODO: is this QLoRA setup wrong?
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import prepare_model_for_kbit_training

    # Set HuggingFace token if available
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # TODO: can just delete these prints right?
    # Load model
    logger.info(f"Loading model {model_name}...")
    if quantization_config is not None:
        logger.info(
            f"  - Mode: QLoRA ({'4-bit' if load_in_4bit else '8-bit'} quantization)"
        )
    else:
        logger.info("  - Mode: LoRA (full precision)")
    logger.info(f"  - Precision: {'bfloat16' if bf16 else 'float16'}")
    logger.info(f"  - Gradient checkpointing: {gradient_checkpointing}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    # CRITICAL: Prepare model based on quantization
    if quantization_config is not None:
        # QLoRA path: Need special preparation for gradient flow through quantized layers
        logger.info("Preparing model for k-bit training (QLoRA)...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    else:
        # LoRA path: Standard gradient checkpointing if requested
        if gradient_checkpointing:
            logger.info("Enabling gradient checkpointing...")
            model.gradient_checkpointing_enable()

    logger.info("✓ Model loaded successfully")
    return model, tokenizer


def apply_lora_to_model(
    model: Any,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[List[str]] = None,
) -> Any:
    """Apply LoRA adapters to a model."""
    # TODO: install peft locally
    from peft import LoraConfig, get_peft_model, TaskType

    # Default target modules if not specified
    # These are common attention/MLP projection layers in transformer models
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            "gate_proj",  # MLP gate (for gated architectures like Llama)
            "up_proj",  # MLP up projection
            "down_proj",  # MLP down projection
        ]

    logger.info(f"Applying LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
    logger.info(f"  - Target modules: {', '.join(lora_target_modules)}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters for verification
    logger.info("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Verify gradient flow is enabled
    trainable_params = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found after applying LoRA! "
            "This should never happen. Check LoRA configuration."
        )

    logger.info(
        f"✓ LoRA adapters applied successfully ({len(trainable_params)} trainable params)"
    )

    return model


def load_lora_checkpoint(
    model: Any,
    checkpoint_path: str,
) -> Any:
    """Load LoRA checkpoint into model using PEFT."""
    from peft import PeftModel

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading LoRA checkpoint from {checkpoint_path}...")

    # Check for PEFT format (adapter_config.json)
    adapter_config = checkpoint_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found at {checkpoint_path}. "
            f"Use PEFT's save_pretrained() to save checkpoints."
        )

    # Get the base model from the PEFT wrapper if already wrapped
    base_model = model.base_model if hasattr(model, "base_model") else model

    # Load adapter weights using PEFT
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        is_trainable=True,
    )
    logger.info("✓ Loaded PEFT checkpoint")

    return model
