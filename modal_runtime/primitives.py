"""Modal primitive functions for distributed training API.

This module provides the core Modal functions that power the training API:
- create_run: Initialize a training run with model + LoRA adapters
- forward_backward: Compute gradients for a batch
- optim_step: Apply optimizer update
- sample: Generate text from the model
- save_state: Export checkpoints and upload to S3

These functions are designed to be called remotely via Modal's function execution system,
allowing for distributed training where gradient computation and optimizer updates
can happen in separate GPU instances.
"""
import os
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

from modal_runtime.app import (
    app,
    TRAINING_IMAGE,
    INFERENCE_IMAGE,
    VOLUME_CONFIG,
    huggingface_secret,
    s3_secret,
    HOURS,
    data_volume,
)
from modal_runtime.model_loader import (
    load_model_and_tokenizer,
    apply_lora_to_model,
    load_lora_checkpoint,
)
from modal_runtime.trainer_utils import (
    setup_optimizer,
    save_optimizer_state,
    load_optimizer_state,
    tokenize_batch,
    compute_forward_backward,
    save_gradients,
    load_gradients,
    save_lora_checkpoint,
    save_merged_model,
    get_run_paths,
    save_run_config,
    load_run_config,
    find_latest_checkpoint,
)
from modal_runtime.gpu_utils import parse_gpu_config
from modal_runtime.manifest import generate_manifest, save_manifest_to_file
from modal_runtime.s3_client import (
    get_artifact_path,
    upload_directory,
    generate_signed_url,
)


def _load_run_model(
    user_id: str,
    run_id: str,
    step: Optional[int] = None,
    for_inference: bool = False,
):
    """Load model and tokenizer for a run with LoRA adapters.
    
    Helper function to consolidate model loading logic across functions.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        step: Optional step to load (if None, loads latest checkpoint)
        for_inference: If True, disable gradient checkpointing for inference
        
    Returns:
        Tuple of (model, tokenizer, config, paths)
    """
    # Load configuration
    config = load_run_config(user_id, run_id)
    paths = get_run_paths(user_id, run_id)
    
    # Load model and tokenizer
    print(f"Loading model {config['base_model']}...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["base_model"],
        load_in_8bit=config.get("load_in_8bit", False),
        load_in_4bit=config.get("load_in_4bit", True),
        max_seq_length=config.get("max_seq_length", 2048),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True) and not for_inference,
    )
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = apply_lora_to_model(
        model=model,
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        lora_target_modules=config.get("lora_target_modules"),
    )
    
    # Load checkpoint if exists
    checkpoint_path = find_latest_checkpoint(paths["lora_adapters"], target_step=step)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model = load_lora_checkpoint(model, str(checkpoint_path))
    elif step is not None and step > 0:
        raise FileNotFoundError(f"Checkpoint for step {step} not found")
    
    return model, tokenizer, config, paths


@app.function(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=2 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden by deployment
)
def create_run(
    user_id: str,
    run_id: str,
    base_model: str,
    framework: str = "transformers",
    gpu_config: str = "l40s:1",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: List[str] = None,
    optimizer: str = "adamw_8bit",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    max_seq_length: int = 2048,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    integrations: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Create a new training run.
    
    Initialize model with LoRA adapters, optimizer, and save initial state.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        base_model: HuggingFace model ID (e.g., 'Qwen/Qwen2.5-3B')
        framework: Framework to use (only 'transformers' supported)
        gpu_config: GPU configuration (e.g., 'l40s:1', 'a100:1')
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA
        optimizer: Optimizer type ('adamw' or 'adamw_8bit')
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_seq_length: Maximum sequence length
        bf16: Use bfloat16 precision
        gradient_checkpointing: Enable gradient checkpointing
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        integrations: Optional dict of integration credentials (wandb, huggingface)
        
    Returns:
        Dict with status, run_id, paths, etc.
    """
    try:
        print("=" * 80)
        print(f"CREATING TRAINING RUN")
        print("=" * 80)
        print(f"User: {user_id}")
        print(f"Run ID: {run_id}")
        print(f"Model: {base_model}")
        print(f"GPU: {gpu_config}")
        print(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print("=" * 80)
        
        # Verify framework
        if framework != "transformers":
            raise ValueError(
                f"Unsupported framework: {framework}. Only 'transformers' is supported. "
                f"Unsloth has been removed for simplicity."
            )
        
        # Parse GPU configuration
        gpu_type, gpu_count = parse_gpu_config(gpu_config)
        
        if gpu_count > 1:
            raise ValueError(
                f"Multi-GPU training is not yet supported. Please use gpu_count=1. "
                f"Got gpu_count={gpu_count} from config '{gpu_config}'"
            )
        
        print(f"GPU: {gpu_count}x {gpu_type}")
        
        # Initialize integrations if provided
        if integrations:
            print("\nSetting up integrations...")
            
            # WandB integration
            if "wandb" in integrations and integrations["wandb"]:
                try:
                    os.environ["WANDB_API_KEY"] = integrations["wandb"]
                    import wandb
                    wandb.init(
                        project="signal-training",
                        name=run_id,
                        config={
                            "base_model": base_model,
                            "lora_r": lora_r,
                            "lora_alpha": lora_alpha,
                            "learning_rate": learning_rate,
                        }
                    )
                    print("✓ WandB initialized")
                except Exception as e:
                    print(f"Warning: Failed to initialize WandB: {e}")
            
            # HuggingFace integration
            if "huggingface" in integrations and integrations["huggingface"]:
                try:
                    os.environ["HF_TOKEN"] = integrations["huggingface"]
                    print("✓ HuggingFace token set")
                except Exception as e:
                    print(f"Warning: Failed to set HuggingFace token: {e}")
        
        # Setup paths
        paths = get_run_paths(user_id, run_id)
        paths["base"].mkdir(parents=True, exist_ok=True)
    
        # Save configuration
        config = {
            "base_model": base_model,
            "framework": framework,
            "gpu_config": gpu_config,
            "gpu_count": gpu_count,
            "gpu_type": gpu_type,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_seq_length": max_seq_length,
            "bf16": bf16,
            "gradient_checkpointing": gradient_checkpointing,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
        }
        save_run_config(user_id, run_id, config)
        
        # Load model and tokenizer
        print("\nLoading model...")
        model, tokenizer = load_model_and_tokenizer(
            model_name=base_model,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # Apply LoRA
        print("\nApplying LoRA adapters...")
        model = apply_lora_to_model(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        
        # Setup optimizer
        print("\nSetting up optimizer...")
        optimizer_instance = setup_optimizer(
            model=model,
            optimizer_type=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    
        # Save initial state
        print("\nSaving initial state...")
        paths["lora_adapters"].mkdir(parents=True, exist_ok=True)
        initial_adapter_path = paths["lora_adapters"] / "step_0"
        save_lora_checkpoint(model, str(initial_adapter_path), tokenizer)
        
        save_optimizer_state(optimizer_instance, str(paths["optimizer_state"]))
        
        # Commit volume
        data_volume.commit()
        
        print("\n" + "=" * 80)
        print(f"✓ RUN CREATED SUCCESSFULLY")
        print("=" * 80)
        
        return {
            "status": "success",
            "run_id": run_id,
            "user_id": user_id,
            "base_model": base_model,
            "framework": framework,
            "paths": {k: str(v) for k, v in paths.items()},
        }
    
    except Exception as e:
        error_msg = f"Run creation failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=2 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden
)
def forward_backward(
    user_id: str,
    run_id: str,
    batch_data: List[Dict[str, Any]],
    step: int,
    accumulate: bool = False,
    loss_fn: str = "causal_lm",
    loss_kwargs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Compute forward and backward pass.
    
    Load model, process batch, compute gradients, and save to volume.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        batch_data: List of training examples
        step: Current training step
        accumulate: Whether to accumulate gradients
        loss_fn: Loss function to use
        loss_kwargs: Additional loss function arguments
        
    Returns:
        Dict with loss, grad_norm, and metrics
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"FORWARD-BACKWARD PASS - Step {step}")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        print(f"Batch size: {len(batch_data)}")
        print(f"Loss function: {loss_fn}")
        
        # Reload volume to ensure we have latest data
        data_volume.reload()
        
        # Load model with LoRA and checkpoint
        model, tokenizer, config, paths = _load_run_model(user_id, run_id, step=step)
        
        # Tokenize batch
        print(f"\nTokenizing {len(batch_data)} examples...")
        batch = tokenize_batch(
            batch_data=batch_data,
            tokenizer=tokenizer,
            max_seq_length=config.get("max_seq_length", 2048),
            loss_fn=loss_fn,
        )
        
        # Compute forward-backward
        print(f"\nComputing forward-backward pass...")
        model.train()
        
        if loss_kwargs is None:
            loss_kwargs = {}
        
        loss, grad_stats = compute_forward_backward(
            model=model,
            batch=batch,
            accumulate=accumulate,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )
        
        # Save gradients
        print("\nSaving gradients...")
        paths["gradients"].mkdir(parents=True, exist_ok=True)
        grad_path = paths["gradients"] / f"step_{step}.pt"
        save_gradients(model, str(grad_path))
        
        # Commit volume
        data_volume.commit()
        
        print(f"\n{'=' * 80}")
        print(f"✓ FORWARD-BACKWARD COMPLETE")
        print(f"Loss: {loss:.4f} | Grad Norm: {grad_stats.get('grad_norm', 0):.4f}")
        print(f"{'=' * 80}")
        
        return {
            "status": "success",
            "loss": loss,
            "step": step,
            "grad_norm": grad_stats.get("grad_norm", 0.0),
            "grad_stats": grad_stats,
        }
    
    except Exception as e:
        error_msg = f"Forward-backward failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=2 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden
)
def optim_step(
    user_id: str,
    run_id: str,
    step: int,
    learning_rate: float = None,
) -> Dict[str, Any]:
    """Apply optimizer update.
    
    Load gradients, update model weights, and save new checkpoint.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        step: Current step (gradients for this step should exist)
        learning_rate: Learning rate override (uses config if None)
        
    Returns:
        Dict with next step number and checkpoint path
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"OPTIMIZER STEP - Step {step}")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        
        # Reload volume to ensure we have latest gradients from forward_backward
        data_volume.reload()
        
        # Load configuration
        config = load_run_config(user_id, run_id)
        paths = get_run_paths(user_id, run_id)
        
        # Use configured learning rate if not provided
        if learning_rate is None:
            learning_rate = config.get("learning_rate", 3e-4)
        
        print(f"Learning rate: {learning_rate}")
        
        # Load model with LoRA and checkpoint
        model, tokenizer, config, paths = _load_run_model(user_id, run_id, step=step)
        
        # Setup optimizer
        print("\nSetting up optimizer...")
        optimizer_instance = setup_optimizer(
            model=model,
            optimizer_type=config.get("optimizer", "adamw_8bit"),
            learning_rate=learning_rate,
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Load optimizer state
        if paths["optimizer_state"].exists():
            print("Loading optimizer state...")
            optimizer_instance = load_optimizer_state(
                optimizer_instance,
                str(paths["optimizer_state"]),
            )
        
        # Load gradients
        grad_path = paths["gradients"] / f"step_{step}.pt"
        if not grad_path.exists():
            raise FileNotFoundError(
                f"Gradients not found for step {step}. "
                f"Make sure to call forward_backward before optim_step."
            )
        
        print(f"Loading gradients from {grad_path}...")
        load_gradients(model, str(grad_path))
        
        # Apply optimizer step
        print("\nApplying optimizer update...")
        optimizer_instance.step()
        optimizer_instance.zero_grad()
        
        # Save updated state
        next_step = step + 1
        print(f"\nSaving checkpoint for step {next_step}...")
        next_checkpoint = paths["lora_adapters"] / f"step_{next_step}"
        save_lora_checkpoint(model, str(next_checkpoint), tokenizer)
        
        save_optimizer_state(optimizer_instance, str(paths["optimizer_state"]))
        
        # Commit volume
        data_volume.commit()
        
        print(f"\n{'=' * 80}")
        print(f"✓ OPTIMIZER STEP COMPLETE")
        print(f"Next step: {next_step}")
        print(f"{'=' * 80}")
        
        return {
            "status": "success",
            "step": next_step,
            "learning_rate": learning_rate,
            "metrics": {
                "checkpoint_path": str(next_checkpoint),
            },
        }
    
    except Exception as e:
        error_msg = f"Optimizer step failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden
)
def sample(
    user_id: str,
    run_id: str,
    prompts: List[str],
    step: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    return_logprobs: bool = False,
) -> Dict[str, Any]:
    """Generate samples from the model.
    
    Load current checkpoint and generate completions.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        prompts: List of prompts to generate from
        step: Step to load checkpoint from
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        return_logprobs: Whether to return log probabilities
        
    Returns:
        Dict with generated outputs
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"GENERATING SAMPLES - Step {step}")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        print(f"Prompts: {len(prompts)}")
        
        # Reload volume to ensure we have latest data
        data_volume.reload()
        
        # Load model with LoRA and checkpoint (for inference)
        model, tokenizer, config, paths = _load_run_model(
            user_id, run_id, step=step, for_inference=True
        )
        model.eval()
        
        # Generate completions
        outputs = []
        all_token_ids = []
        all_tokens = []
        all_logprobs = [] if return_logprobs else None
        
        print(f"\nGenerating with temperature={temperature}, top_p={top_p}...")
        
        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                if return_logprobs:
                    # Generate with scores for logprobs
                    generation_output = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    generated_ids = generation_output.sequences[0]
                    scores = generation_output.scores  # Tuple of tensors, one per generated token
                    
                    # Compute logprobs from scores
                    import torch.nn.functional as F
                    token_logprobs = []
                    for score in scores:
                        # score is shape (batch_size, vocab_size)
                        log_probs = F.log_softmax(score[0], dim=-1)
                        # Get the logprob of the selected token
                        # We need to get which token was actually selected
                        # The scores correspond to the generated tokens after input
                        token_logprobs.append(log_probs.cpu().tolist())
                    
                    # For simpler output, just return the logprob of the selected tokens
                    selected_token_ids = generated_ids[input_length:].cpu().tolist()
                    selected_logprobs = []
                    for idx, token_id in enumerate(selected_token_ids):
                        if idx < len(token_logprobs):
                            selected_logprobs.append(token_logprobs[idx][token_id])
                    
                    all_logprobs.append(selected_logprobs)
                else:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )[0]
            
            # Decode full output
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs.append(output_text)
            
            # Extract generated tokens (excluding prompt)
            generated_token_ids = generated_ids[input_length:].cpu().tolist()
            all_token_ids.append(generated_token_ids)
            
            # Convert token IDs to token strings
            token_strings = tokenizer.convert_ids_to_tokens(generated_token_ids)
            all_tokens.append(token_strings)
            
            print(f"    Output: {output_text[:100]}...")
            print(f"    Tokens generated: {len(generated_token_ids)}")
        
        print(f"\n{'=' * 80}")
        print(f"✓ GENERATED {len(outputs)} COMPLETIONS")
        print(f"{'=' * 80}")
        
        return {
            "status": "success",
            "outputs": outputs,
            "token_ids": all_token_ids,
            "tokens": all_tokens,
            "logprobs": all_logprobs,
        }
    
    except Exception as e:
        error_msg = f"Sampling failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden
)
def sample_stream(
    user_id: str,
    run_id: str,
    prompt: str,
    step: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate samples from the model with streaming (token-by-token).
    
    Load current checkpoint and stream completions token by token.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        prompt: Single prompt to generate from
        step: Step to load checkpoint from
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        
    Yields:
        Dict with token, token_id, logprob, and is_finished flag
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"STREAMING GENERATION - Step {step}")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        print(f"Prompt: {prompt[:50]}...")
        
        # Reload volume to ensure we have latest data
        data_volume.reload()
        
        # Load model with LoRA and checkpoint (for inference)
        model, tokenizer, config, paths = _load_run_model(
            user_id, run_id, step=step, for_inference=True
        )
        model.eval()
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs['input_ids'].shape[1]
        
        print(f"Generating with temperature={temperature}, top_p={top_p}...")
        
        # Use generate with streaming
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Run generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens as they're generated
        for token_text in streamer:
            # Tokenize to get token_id (approximation)
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            token_id = token_ids[0] if token_ids else 0
            
            yield {
                "token": token_text,
                "token_id": token_id,
                "logprob": None,  # Not easily available in streaming mode
                "is_finished": False,
            }
        
        # Wait for generation to complete
        thread.join()
        
        # Send finish signal
        yield {
            "token": "",
            "token_id": 0,
            "logprob": None,
            "is_finished": True,
        }
        
        print(f"\n{'=' * 80}")
        print(f"✓ STREAMING GENERATION COMPLETE")
        print(f"{'=' * 80}")
    
    except Exception as e:
        error_msg = f"Streaming failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden
)
def embeddings(
    user_id: str,
    run_id: str,
    texts: List[str],
    step: int,
    layer: int = -1,
    pooling: str = "mean",
) -> Dict[str, Any]:
    """Generate embeddings from the model.
    
    Extract hidden states from specified layer and apply pooling.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        texts: List of texts to embed
        step: Step to load checkpoint from
        layer: Layer to extract embeddings from (-1 for last layer)
        pooling: Pooling strategy ('mean', 'last_token', 'cls_token')
        
    Returns:
        Dict with embeddings list and dimensions
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"GENERATING EMBEDDINGS - Step {step}")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        print(f"Texts: {len(texts)}")
        print(f"Layer: {layer}, Pooling: {pooling}")
        
        # Reload volume to ensure we have latest data
        data_volume.reload()
        
        # Load model with LoRA and checkpoint (for inference)
        model, tokenizer, config, paths = _load_run_model(
            user_id, run_id, step=step, for_inference=True
        )
        model.eval()
        
        all_embeddings = []
        
        for i, text in enumerate(texts):
            print(f"  Embedding {i+1}/{len(texts)}: {text[:50]}...")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # Get hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
                
                # Select layer
                if layer == -1:
                    layer_hidden_states = hidden_states[-1]  # Last layer
                else:
                    layer_hidden_states = hidden_states[layer]
                
                # Apply pooling
                if pooling == "mean":
                    # Mean pooling over sequence length
                    attention_mask = inputs.get('attention_mask')
                    if attention_mask is not None:
                        # Mask out padding tokens
                        mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
                        sum_embeddings = torch.sum(layer_hidden_states * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        embedding = sum_embeddings / sum_mask
                    else:
                        embedding = torch.mean(layer_hidden_states, dim=1)
                    
                elif pooling == "last_token":
                    # Use last non-padding token
                    attention_mask = inputs.get('attention_mask')
                    if attention_mask is not None:
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        batch_size = layer_hidden_states.shape[0]
                        embedding = layer_hidden_states[torch.arange(batch_size, device=layer_hidden_states.device), sequence_lengths]
                    else:
                        embedding = layer_hidden_states[:, -1, :]
                    
                elif pooling == "cls_token":
                    # Use first token (CLS token)
                    embedding = layer_hidden_states[:, 0, :]
                    
                else:
                    raise ValueError(f"Invalid pooling strategy: {pooling}")
                
                # Convert to list
                embedding_list = embedding[0].cpu().tolist()
                all_embeddings.append(embedding_list)
        
        dimensions = len(all_embeddings[0]) if all_embeddings else 0
        
        print(f"\n{'=' * 80}")
        print(f"✓ GENERATED {len(all_embeddings)} EMBEDDINGS")
        print(f"Dimensions: {dimensions}")
        print(f"{'=' * 80}")
        
        return {
            "status": "success",
            "embeddings": all_embeddings,
            "dimensions": dimensions,
        }
    
    except Exception as e:
        error_msg = f"Embeddings generation failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    timeout=2 * HOURS,
    gpu="l40s:1",  # Default GPU, will be overridden
)
def save_state(
    user_id: str,
    run_id: str,
    step: int,
    mode: str = "adapter",
    push_to_hub: bool = False,
    hub_model_id: str = None,
    training_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save model state with auto-export to S3.
    
    Export LoRA adapter or merged model to Modal Volume and S3.
    Optionally push to HuggingFace Hub.
    
    Args:
        user_id: User identifier
        run_id: Run identifier  
        step: Training step number
        mode: Export mode ('adapter', 'merged', or 'state')
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub repository ID
        training_metrics: Optional dict of training metrics
        
    Returns:
        Dict with artifact paths, S3 URI, and signed download URL
    """
    from datetime import datetime, timezone, timedelta
    
    try:
        print(f"\n{'=' * 80}")
        print(f"SAVING STATE - Step {step}")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        print(f"Mode: {mode}")
        
        # Get paths and config
        config = load_run_config(user_id, run_id)
        paths = get_run_paths(user_id, run_id)
        
        # Find checkpoint
        checkpoint_path = find_latest_checkpoint(paths["lora_adapters"], target_step=step)
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoints found for run {run_id}")
        
        # Create export directory
        paths["checkpoints"].mkdir(parents=True, exist_ok=True)
        export_path = paths["checkpoints"] / f"step_{step}_{mode}"
        
        # Save to Modal Volume based on mode
        if mode == "adapter":
            # Just copy the adapter
            import shutil
            if export_path.exists():
                shutil.rmtree(export_path)
            shutil.copytree(checkpoint_path, export_path)
            artifact_uri = str(export_path)
            print(f"✓ Copied adapter to {export_path}")
        
        elif mode == "merged":
            # Load model for merging
            model, tokenizer, _, _ = _load_run_model(user_id, run_id, step=step, for_inference=True)
            
            # Save merged model
            save_merged_model(
                model=model,
                base_model_name=config["base_model"],
                save_path=str(export_path),
                tokenizer=tokenizer,
                framework=config["framework"],
            )
            artifact_uri = str(export_path)
        
        elif mode == "state":
            # Full training state: LoRA + optimizer + gradients
            import shutil
            if export_path.exists():
                shutil.rmtree(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Copy LoRA adapters
            shutil.copytree(checkpoint_path, export_path / "lora_adapters")
            
            # Copy optimizer state
            if paths["optimizer_state"].exists():
                shutil.copy2(paths["optimizer_state"], export_path / "optimizer_state.pt")
            
            # Copy latest gradient
            grad_path = paths["gradients"] / f"step_{step}.pt"
            if grad_path.exists():
                (export_path / "gradients").mkdir(exist_ok=True)
                shutil.copy2(grad_path, export_path / "gradients" / f"step_{step}.pt")
            
            artifact_uri = str(export_path)
            print(f"✓ Saved full state to {export_path}")
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'adapter', 'merged', or 'state'")
        
        # Commit volume immediately
        data_volume.commit()
        print(f"✓ Committed to Modal Volume")
        
        # Generate manifest
        print("\nGenerating manifest...")
        manifest = generate_manifest(
            run_id=run_id,
            owner_id=user_id,
            step=step,
            run_config=config,
            artifact_path=str(export_path),
            mode=mode,
            metrics=training_metrics,
        )
        
        # Save manifest to local directory
        manifest_path = export_path / "manifest.json"
        save_manifest_to_file(manifest, str(manifest_path))
        
        # Upload to S3
        s3_uri = None
        download_url = None
        download_expires_at = None
        
        try:
            print("\nUploading to S3...")
            s3_prefix = get_artifact_path(user_id, run_id, mode, step)
            
            upload_result = upload_directory(
                local_path=str(export_path),
                s3_prefix=s3_prefix,
            )
            
            s3_uri = upload_result["s3_uri"]
            print(f"✓ Uploaded to S3: {s3_uri}")
            
            # Generate signed download URL (1 hour expiration)
            download_url = generate_signed_url(s3_uri, expiration=3600)
            download_expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            
            print(f"✓ Generated signed URL (expires in 1 hour)")
            
        except Exception as e:
            print(f"⚠ Warning: Failed to upload to S3: {e}")
            print("Artifact is still available on Modal Volume")
        
        # Push to HuggingFace Hub if requested
        if push_to_hub and hub_model_id:
            print(f"\nPushing to HuggingFace Hub: {hub_model_id}")
            from huggingface_hub import HfApi
            
            api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
            api.upload_folder(
                folder_path=str(export_path),
                repo_id=hub_model_id,
                repo_type="model",
            )
            print(f"✓ Pushed to HuggingFace Hub")
        
        print(f"\n{'=' * 80}")
        print(f"✓ STATE SAVED SUCCESSFULLY")
        print(f"{'=' * 80}")
        
        return {
            "status": "success",
            "artifact_uri": artifact_uri,
            "local_path": str(export_path),
            "s3_uri": s3_uri,
            "download_url": download_url,
            "download_expires_at": download_expires_at,
            "manifest": manifest,
            "pushed_to_hub": push_to_hub,
            "hub_model_id": hub_model_id if push_to_hub else None,
        }
    
    except Exception as e:
        error_msg = f"Save state failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=10 * 60,  # 10 minutes
    gpu="l40s:1",
)
def tokenize(
    user_id: str,
    run_id: str,
    text: str | List[str],
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    """Tokenize text using the model's tokenizer.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        text: Text string or list of strings to tokenize
        add_special_tokens: Whether to add special tokens
        
    Returns:
        Dict with token_ids and token strings
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"TOKENIZING TEXT")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        
        # Reload volume
        data_volume.reload()
        
        # Load tokenizer (no need to load full model for tokenization)
        _, tokenizer, _, _ = _load_run_model(
            user_id, run_id, step=0, for_inference=True
        )
        
        # Handle single string or list of strings
        texts = [text] if isinstance(text, str) else text
        
        # Tokenize
        encoded = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            return_tensors=None,  # Return lists instead of tensors
        )
        
        # Convert to token strings
        all_tokens = []
        for token_ids in encoded['input_ids']:
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            all_tokens.append(tokens)
        
        print(f"✓ Tokenized {len(texts)} text(s)")
        
        return {
            "status": "success",
            "token_ids": encoded['input_ids'],
            "tokens": all_tokens,
        }
    
    except Exception as e:
        error_msg = f"Tokenization failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=10 * 60,  # 10 minutes
    gpu="l40s:1",
)
def detokenize(
    user_id: str,
    run_id: str,
    token_ids: List[int] | List[List[int]],
) -> Dict[str, Any]:
    """Detokenize token IDs using the model's tokenizer.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        token_ids: Token IDs (single list or list of lists)
        
    Returns:
        Dict with decoded text
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"DETOKENIZING TOKEN IDS")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        
        # Reload volume
        data_volume.reload()
        
        # Load tokenizer
        _, tokenizer, _, _ = _load_run_model(
            user_id, run_id, step=0, for_inference=True
        )
        
        # Handle single list or list of lists
        is_single = isinstance(token_ids[0], int) if token_ids else True
        ids_to_decode = [token_ids] if is_single else token_ids
        
        # Decode
        texts = []
        for ids in ids_to_decode:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        
        result = texts[0] if is_single else texts
        
        print(f"✓ Detokenized {len(ids_to_decode)} sequence(s)")
        
        return {
            "status": "success",
            "text": result,
        }
    
    except Exception as e:
        error_msg = f"Detokenization failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=10 * 60,  # 10 minutes
    gpu="l40s:1",
)
def get_tokenizer_info(
    user_id: str,
    run_id: str,
) -> Dict[str, Any]:
    """Get tokenizer configuration information.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        
    Returns:
        Dict with tokenizer info
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"GETTING TOKENIZER INFO")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        
        # Reload volume
        data_volume.reload()
        
        # Load tokenizer
        _, tokenizer, _, _ = _load_run_model(
            user_id, run_id, step=0, for_inference=True
        )
        
        # Extract special tokens
        special_tokens = {}
        if tokenizer.bos_token:
            special_tokens['bos_token'] = tokenizer.bos_token
        if tokenizer.eos_token:
            special_tokens['eos_token'] = tokenizer.eos_token
        if tokenizer.pad_token:
            special_tokens['pad_token'] = tokenizer.pad_token
        if tokenizer.unk_token:
            special_tokens['unk_token'] = tokenizer.unk_token
        if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token:
            special_tokens['sep_token'] = tokenizer.sep_token
        if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token:
            special_tokens['cls_token'] = tokenizer.cls_token
        
        info = {
            "status": "success",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else None,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None,
            "special_tokens": special_tokens,
        }
        
        print(f"✓ Retrieved tokenizer info")
        print(f"  Vocab size: {info['vocab_size']}")
        print(f"  Special tokens: {list(special_tokens.keys())}")
        
        return info
    
    except Exception as e:
        error_msg = f"Get tokenizer info failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=10 * 60,  # 10 minutes
    gpu="l40s:1",
)
def get_model_info(
    user_id: str,
    run_id: str,
) -> Dict[str, Any]:
    """Get model architecture information.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        
    Returns:
        Dict with model info
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"GETTING MODEL INFO")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        
        # Reload volume
        data_volume.reload()
        
        # Load model and tokenizer
        model, tokenizer, config, _ = _load_run_model(
            user_id, run_id, step=0, for_inference=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get architecture details from config
        model_config = model.config
        architecture = model_config.architectures[0] if hasattr(model_config, 'architectures') else type(model).__name__
        
        # Get chat template if available
        chat_template = None
        if hasattr(tokenizer, 'chat_template'):
            chat_template = tokenizer.chat_template
        
        info = {
            "status": "success",
            "base_model": config.get("base_model", "unknown"),
            "architecture": architecture,
            "num_parameters": total_params,
            "num_trainable_parameters": trainable_params,
            "hidden_size": getattr(model_config, 'hidden_size', None),
            "num_layers": getattr(model_config, 'num_hidden_layers', None),
            "num_attention_heads": getattr(model_config, 'num_attention_heads', None),
            "vocab_size": getattr(model_config, 'vocab_size', None),
            "max_position_embeddings": getattr(model_config, 'max_position_embeddings', None),
            "chat_template": chat_template,
        }
        
        print(f"✓ Retrieved model info")
        print(f"  Architecture: {architecture}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return info
    
    except Exception as e:
        error_msg = f"Get model info failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise


@app.function(
    image=INFERENCE_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=10 * 60,  # 10 minutes
    gpu="l40s:1",
)
def apply_chat_template(
    user_id: str,
    run_id: str,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> Dict[str, Any]:
    """Apply the model's chat template to format messages.
    
    Args:
        user_id: User identifier
        run_id: Run identifier
        messages: List of message dicts with 'role' and 'content'
        add_generation_prompt: Whether to add generation prompt
        
    Returns:
        Dict with formatted text and token_ids
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"APPLYING CHAT TEMPLATE")
        print(f"{'=' * 80}")
        print(f"Run: {run_id}")
        print(f"Messages: {len(messages)}")
        
        # Reload volume
        data_volume.reload()
        
        # Load tokenizer
        _, tokenizer, _, _ = _load_run_model(
            user_id, run_id, step=0, for_inference=True
        )
        
        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            # Get formatted text
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            
            # Also get tokenized version
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Fallback: simple concatenation
            print("⚠ Warning: Tokenizer doesn't have chat_template, using simple format")
            parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                parts.append(f"{role.capitalize()}: {content}")
            
            formatted_text = "\n".join(parts)
            if add_generation_prompt:
                formatted_text += "\nAssistant:"
            
            token_ids = tokenizer.encode(formatted_text)
        
        print(f"✓ Applied chat template")
        print(f"  Formatted length: {len(formatted_text)} chars")
        print(f"  Token count: {len(token_ids)}")
        
        return {
            "status": "success",
            "text": formatted_text,
            "token_ids": token_ids,
        }
    
    except Exception as e:
        error_msg = f"Apply chat template failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ ERROR: {error_msg}")
        raise
