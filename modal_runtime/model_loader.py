"""Model loading utilities for different frameworks."""
import os
import torch
from pathlib import Path
from typing import Tuple, Optional, Any


def load_model_and_tokenizer(
    model_name: str,
    framework: str = "transformers",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    max_seq_length: int = 2048,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any]:
    """Load model and tokenizer based on framework.
    
    Args:
        model_name: HuggingFace model ID
        framework: Framework to use ('transformers' or 'unsloth')
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        max_seq_length: Maximum sequence length
        bf16: Use bfloat16 precision
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Tuple of (model, tokenizer/processor)
    """
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    
    if framework == "unsloth":
        return _load_unsloth_model(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        return _load_transformers_model(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
        )


def _load_transformers_model(
    model_name: str,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any]:
    """Load model using Transformers library.
    
    Args:
        model_name: HuggingFace model ID
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        bf16: Use bfloat16 precision
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def _load_unsloth_model(
    model_name: str,
    load_in_4bit: bool = False,
    max_seq_length: int = 2048,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any]:
    """Load model using Unsloth library.
    
    Args:
        model_name: HuggingFace model ID or Unsloth model ID
        load_in_4bit: Use 4-bit quantization
        max_seq_length: Maximum sequence length
        bf16: Use bfloat16 precision
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
    )
    
    return model, tokenizer


def apply_lora_to_model(
    model: Any,
    framework: str = "transformers",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[list] = None,
) -> Any:
    """Apply LoRA adapters to a model.
    
    Args:
        model: Base model
        framework: Framework used ('transformers' or 'unsloth')
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        
    Returns:
        Model with LoRA adapters
    """
    if framework == "unsloth":
        return _apply_lora_unsloth(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    else:
        return _apply_lora_peft(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )


def _apply_lora_peft(
    model: Any,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[list] = None,
) -> Any:
    """Apply LoRA using PEFT library.
    
    Args:
        model: Base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        
    Returns:
        Model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Default target modules if not specified
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def _apply_lora_unsloth(
    model: Any,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
) -> Any:
    """Apply LoRA using Unsloth library.
    
    Args:
        model: Base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        
    Returns:
        Model with LoRA adapters
    """
    from unsloth import FastLanguageModel
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    return model


def load_lora_checkpoint(
    model: Any,
    checkpoint_path: str,
) -> Any:
    """Load LoRA checkpoint into model.
    
    Args:
        model: Model with LoRA adapters
        checkpoint_path: Path to LoRA checkpoint
        
    Returns:
        Model with loaded checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load adapter weights
    from peft import PeftModel
    
    try:
        # Try loading as PEFT model
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
    except:
        # If that fails, try loading state dict directly
        state_dict = torch.load(checkpoint_path / "adapter_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model

