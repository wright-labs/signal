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
        
        # Load model with LoRA and checkpoint (for inference)
        model, tokenizer, config, paths = _load_run_model(
            user_id, run_id, step=step, for_inference=True
        )
        model.eval()
        
        # Generate completions
        outputs = []
        print(f"\nGenerating with temperature={temperature}, top_p={top_p}...")
        
        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(output_text)
            print(f"    Output: {output_text[:100]}...")
        
        print(f"\n{'=' * 80}")
        print(f"✓ GENERATED {len(outputs)} COMPLETIONS")
        print(f"{'=' * 80}")
        
        return {
            "status": "success",
            "outputs": outputs,
            "logprobs": None if not return_logprobs else [],  # TODO: Implement logprobs
        }
    
    except Exception as e:
        error_msg = f"Sampling failed: {str(e)}\n{traceback.format_exc()}"
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
