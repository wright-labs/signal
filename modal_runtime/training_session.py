"""Stateful training session with Accelerate for multi-GPU support."""

import torch
import modal
from typing import Dict, Any, List, Optional
import time
import threading
import traceback
import logging

from accelerate import Accelerator

logger = logging.getLogger(__name__)

from modal_runtime.app import (
    app,
    TRAINING_IMAGE,
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
from modal_runtime.utils import (
    tokenize_batch,
    get_run_paths,
    save_run_config,
    find_latest_checkpoint,
)
from modal_runtime.gpu_monitor import (
    get_gpu_summary,
)


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="L40S:2",  # Can be overridden at runtime
    timeout=2 * HOURS,
    scaledown_window=20 * 60,  # 20 minutes idle before shutdown
)
@modal.concurrent(max_inputs=10)  # Handle multiple requests to same container
class TrainingSession:
    """Stateful training session with Accelerate for multi-GPU."""

    model: Any = None
    tokenizer: Any = None
    optimizer: Any = None
    scheduler: Any = None
    accelerator: Accelerator = None
    user_id: str = None
    run_id: str = None
    config: Dict[str, Any] = None
    current_step: int = 0
    last_checkpoint_step: int = 0
    auto_checkpoint_interval: int = 100
    accumulation_count: int = 0
    accumulation_steps: int = 1
    last_activity_time: float = None
    should_monitor: bool = False
    monitor_thread: threading.Thread = None
    wandb_run: Any = None
    last_loss: float = 0.0

    @modal.enter()
    def container_startup(self):
        self.accelerator = Accelerator(
            mixed_precision="bf16",  # Use bfloat16 for training
            gradient_accumulation_steps=1,  # Will be set in initialize()
        )

        # Detect GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("⚠️  No CUDA GPUs detected!")

        logger.info(
            f"✓ Accelerator initialized (num_processes: {self.accelerator.num_processes})"
        )

        # Start background monitoring thread for auto-checkpoint
        self.should_monitor = True
        self.monitor_thread = threading.Thread(
            target=self._background_monitor, daemon=True
        )
        self.monitor_thread.start()
        logger.info("✓ Background monitoring thread started")

    @modal.exit()
    def container_shutdown(self):
        """Called when container shuts down.

        This runs when:
        - Container idle timeout (20 minutes)
        - Manual shutdown
        - Container crashes
        """

        logger.info("CONTAINER SHUTTING DOWN")

        # Stop monitoring thread
        self.should_monitor = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Auto-save if model is loaded
        if self.model is not None:
            try:
                logger.info(f"Auto-saving checkpoint at step {self.current_step}...")
                self._save_checkpoint_internal(tag=f"autosave-{int(time.time())}")
                data_volume.commit()
                logger.info("✓ Auto-save complete")
            except Exception as e:
                logger.warning(f"⚠ Warning: Auto-save failed: {e}")

        # Clean up GPU memory
        if self.model is not None:
            del self.model
        if self.optimizer is not None:
            del self.optimizer

        torch.cuda.empty_cache()
        logger.info("✓ GPU memory cleaned up")

        # Finish WandB run if active
        if self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("✓ WandB run finished")
            except Exception as e:
                logger.warning(f"⚠ Failed to finish WandB run: {e}")

        logger.info("✓ Container shutdown complete")

    # TODO: check where this is called and if base_model names is validated from the set
    @modal.method()
    def initialize(
        self,
        user_id: str,
        run_id: str,
        base_model: str,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        optimizer: str = "adamw_8bit",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        bf16: bool = True,
        gradient_checkpointing: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        resume_from_step: Optional[int] = None,
        accumulation_steps: int = 1,
        auto_checkpoint_interval: int = 100,
        integrations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Initialize training session.

        This loads the model into GPU memory and keeps it there.
        All subsequent calls will reuse this loaded model.

        Args:
            user_id: User identifier
            run_id: Run identifier
            base_model: HuggingFace model ID
            resume_from_step: If provided, resume from this checkpoint
            accumulation_steps: Gradient accumulation steps
            ... (other hyperparameters)

        Returns:
            Dict with session info and model stats
        """
        try:
            self._update_activity()
            logger.info("INITIALIZING TRAINING SESSION")

            logger.info(f"User: {user_id}")
            logger.info(f"Run: {run_id}")
            logger.info(f"Model: {base_model}")
            logger.info(f"Resume from step: {resume_from_step}")

            # Store session info
            self.user_id = user_id
            self.run_id = run_id
            self.accumulation_steps = accumulation_steps
            self.auto_checkpoint_interval = auto_checkpoint_interval

            # Update accelerator's gradient accumulation
            self.accelerator.gradient_accumulation_steps = accumulation_steps

            # Reload volume to get latest checkpoints
            data_volume.reload()

            # Setup paths
            paths = get_run_paths(user_id, run_id)
            paths["base"].mkdir(parents=True, exist_ok=True)

            # Store configuration
            self.config = {
                "base_model": base_model,
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
                "accumulation_steps": accumulation_steps,
                "auto_checkpoint_interval": auto_checkpoint_interval,
            }

            # Initialize WandB if credentials provided
            if integrations and integrations.get("wandb"):
                try:
                    import wandb
                    import os
                    from datetime import datetime

                    # Create experiment name: model_datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    experiment_name = f"{base_model.replace('/', '_')}_{timestamp}"

                    # Initialize WandB
                    os.environ["WANDB_API_KEY"] = integrations["wandb"]
                    self.wandb_run = wandb.init(
                        project="signal-training",
                        name=experiment_name,
                        config=self.config,
                        resume="allow",
                        id=run_id,  # Use run_id for resume capability
                    )
                    logger.info(f"✓ WandB initialized: {experiment_name}")
                except Exception as e:
                    logger.warning(f"⚠ Failed to initialize WandB: {e}")
                    self.wandb_run = None
            else:
                self.wandb_run = None
                logger.info("ℹ WandB not configured (no API key)")

            # Check if resuming from checkpoint
            if resume_from_step is not None:
                checkpoint_path = find_latest_checkpoint(
                    paths["lora_adapters"], target_step=resume_from_step
                )
                if not checkpoint_path:
                    raise FileNotFoundError(
                        f"Checkpoint for step {resume_from_step} not found"
                    )

                logger.info(f"\nResuming from checkpoint: {checkpoint_path}")
                self.current_step = resume_from_step
                self.last_checkpoint_step = resume_from_step
            else:
                logger.info("\nStarting fresh training run")
                self.current_step = 0
                self.last_checkpoint_step = 0

            # Load model and tokenizer (THIS IS THE EXPENSIVE PART - 30-60s)
            logger.info("\nLoading model and tokenizer...")
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=base_model,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                max_seq_length=max_seq_length,
                bf16=bf16,
                gradient_checkpointing=gradient_checkpointing,
                device_map="auto",  # Let Accelerate handle device placement
            )

            # Apply LoRA adapters
            logger.info("\nApplying LoRA adapters...")
            self.model = apply_lora_to_model(
                model=self.model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
            )

            # Load checkpoint if resuming
            if resume_from_step is not None:
                logger.info(f"\nLoading checkpoint from step {resume_from_step}...")
                checkpoint_path = find_latest_checkpoint(
                    paths["lora_adapters"], target_step=resume_from_step
                )
                self.model = load_lora_checkpoint(self.model, str(checkpoint_path))

            # Setup optimizer (inline from deleted utils/optimizer.py)
            logger.info(f"\nSetting up {optimizer} optimizer...")
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

            if optimizer == "adamw_8bit":
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            elif optimizer == "adamw":
                from torch.optim import AdamW

                self.optimizer = AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

            # Prepare model and optimizer with Accelerate (handles multi-GPU)
            logger.info("\nPreparing model and optimizer with Accelerate...")
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer
            )

            # Load optimizer state if resuming
            if resume_from_step is not None:
                opt_state_path = paths["optimizer_state"]
                if opt_state_path.exists():
                    logger.info("Loading optimizer state...")
                    state_dict = torch.load(opt_state_path, map_location="cpu")
                    self.optimizer.load_state_dict(state_dict)

            # TODO: Add scheduler support
            # self.scheduler = get_cosine_schedule_with_warmup(...)

            # Save configuration
            save_run_config(user_id, run_id, self.config)

            # Save initial checkpoint if fresh start
            if resume_from_step is None:
                logger.info("\nSaving initial checkpoint (step 0)...")
                self._save_checkpoint_internal(tag="initial")

            # Commit volume
            data_volume.commit()

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            logger.info("✓ INITIALIZATION COMPLETE")

            logger.info("Model loaded and ready in GPU memory")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Current step: {self.current_step}")
            logger.info(f"Num processes: {self.accelerator.num_processes}")

            return {
                "status": "success",
                "run_id": run_id,
                "user_id": user_id,
                "current_step": self.current_step,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "num_processes": self.accelerator.num_processes,
                "config": self.config,
            }

        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}\n{traceback.format_exc()}"
            logger.info(f"\n❌ ERROR: {error_msg}")
            raise

    @modal.method()
    def forward_backward(
        self,
        batch_data: List[Dict[str, Any]],
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute forward and backward pass.

        This uses the already-loaded model from initialize().
        No model loading happens here - just pure training!

        Returns:
            Dict with loss, metrics, and current step
        """
        try:
            self._update_activity()

            if self.model is None:
                raise RuntimeError("Model not initialized. Call initialize() first.")

            logger.info("FORWARD-BACKWARD PASS")

            logger.info(f"Step: {self.current_step}")
            logger.info(f"Batch size: {len(batch_data)}")
            logger.info(f"Loss function: {loss_fn}")

            # Tokenize batch
            batch = tokenize_batch(
                batch_data=batch_data,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.get("max_seq_length", 2048),
                loss_fn=loss_fn,
            )

            # Set model to training mode
            self.model.train()

            # Move batch to device (Accelerate handles this)
            batch = {
                k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if loss_kwargs is None:
                loss_kwargs = {}

            # FORWARD PASS
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels", batch["input_ids"]),
                )

                # Get loss from model outputs (HuggingFace models return loss)
                loss = outputs.loss

                # Ensure loss is scalar
                if loss.dim() > 0:
                    loss = loss.mean()

                # BACKWARD PASS (Accelerate handles gradient accumulation)
                self.accelerator.backward(loss)

            # Compute gradient statistics
            grad_norm = 0.0
            if self.accelerator.sync_gradients:
                # Only compute grad norm when gradients are synchronized
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    float("inf"),  # No clipping, just compute norm
                )

            # Track gradient accumulation
            self.accumulation_count += 1
            if self.accumulation_count >= self.accumulation_steps:
                self.accumulation_count = 0

            metrics = {
                "loss": loss.item(),
                "perplexity": torch.exp(loss).item(),
                "grad_norm": grad_norm
                if isinstance(grad_norm, float)
                else grad_norm.item(),
            }

            # Store loss for WandB logging in optim_step
            self.last_loss = metrics["loss"]

            logger.info("✓ FORWARD-BACKWARD COMPLETE")
            logger.info(
                f"Loss: {metrics['loss']:.4f} | Grad Norm: {metrics['grad_norm']:.4f}"
            )
            logger.info(
                f"Accumulation: {self.accumulation_count}/{self.accumulation_steps}"
            )

            return {
                "status": "success",
                "loss": metrics["loss"],
                "step": self.current_step,
                "accumulation_count": self.accumulation_count,
                "metrics": metrics,
            }

        except Exception as e:
            error_msg = f"Forward-backward failed: {str(e)}\n{traceback.format_exc()}"
            logger.info(f"\n❌ ERROR: {error_msg}")
            raise

    @modal.method()
    def optim_step(
        self,
        learning_rate: Optional[float] = None,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply optimizer update.

        This uses the already-loaded optimizer from initialize().
        No optimizer creation or loading happens here - just the update!

        Returns:
            Dict with new step number and metrics
        """
        try:
            self._update_activity()

            if self.model is None or self.optimizer is None:
                raise RuntimeError(
                    "Model/optimizer not initialized. Call initialize() first."
                )

            logger.info("OPTIMIZER STEP")

            logger.info(f"Current step: {self.current_step}")

            # Override learning rate if provided
            if learning_rate is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = learning_rate

            # Clip gradients if requested (Accelerate-aware)
            if grad_clip is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Apply optimizer step (only if gradients are synchronized)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()

            # Increment step counter
            self.current_step += 1

            # Check if auto-checkpoint needed
            steps_since_checkpoint = self.current_step - self.last_checkpoint_step
            checkpoint_saved = False
            if steps_since_checkpoint >= self.auto_checkpoint_interval:
                logger.info(
                    f"\nAuto-checkpoint triggered (every {self.auto_checkpoint_interval} steps)"
                )
                self._save_checkpoint_internal()
                data_volume.commit()
                checkpoint_saved = True

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log to WandB if initialized
            if self.wandb_run:
                try:
                    self.wandb_run.log(
                        {
                            "train/loss": self.last_loss,
                            "train/learning_rate": current_lr,
                            "train/step": self.current_step,
                        }
                    )
                except Exception as e:
                    logger.info(f"⚠ Failed to log to WandB: {e}")

            logger.info("✓ OPTIMIZER STEP COMPLETE")
            logger.info(f"New step: {self.current_step}")
            logger.info(f"Learning rate: {current_lr}")

            return {
                "status": "success",
                "step": self.current_step,
                "learning_rate": current_lr,
                "checkpoint_saved": checkpoint_saved,
            }

        except Exception as e:
            error_msg = f"Optimizer step failed: {str(e)}\n{traceback.format_exc()}"
            logger.info(f"\n❌ ERROR: {error_msg}")
            raise

    @modal.method()
    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        return_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Generate text samples."""
        try:
            self._update_activity()

            if self.model is None:
                raise RuntimeError("Model not initialized. Call initialize() first.")

            logger.info("GENERATING SAMPLES")

            logger.info(f"Step: {self.current_step}")
            logger.info(f"Prompts: {len(prompts)}")

            # Unwrap model for generation (Accelerate wraps it)
            model_to_use = self.accelerator.unwrap_model(self.model)

            # Switch to eval mode
            model_to_use.eval()

            outputs = []
            all_token_ids = []
            all_logprobs = [] if return_logprobs else None

            with torch.no_grad():
                for i, prompt in enumerate(prompts):
                    logger.info(f"  Generating {i + 1}/{len(prompts)}...")

                    # Tokenize prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(
                        self.accelerator.device
                    )

                    # Generate
                    if return_logprobs:
                        generation_output = model_to_use.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=temperature > 0,
                            pad_token_id=self.tokenizer.pad_token_id,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        generated_ids = generation_output.sequences[0]
                        # TODO: Extract logprobs from scores
                    else:
                        generated_ids = model_to_use.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=temperature > 0,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )[0]

                    # Decode
                    output_text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    outputs.append(output_text)
                    all_token_ids.append(generated_ids.cpu().tolist())

            # Switch back to train mode
            model_to_use.train()

            logger.info(f"✓ GENERATED {len(outputs)} COMPLETIONS")

            return {
                "status": "success",
                "outputs": outputs,
                "token_ids": all_token_ids,
                "logprobs": all_logprobs,
                "step": self.current_step,
            }

        except Exception as e:
            error_msg = f"Sampling failed: {str(e)}\n{traceback.format_exc()}"
            logger.info(f"\n❌ ERROR: {error_msg}")
            raise

    @modal.method()
    def save_state(
        self,
        mode: str = "adapter",
        tag: Optional[str] = None,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model state to volume and S3.

        Args:
            mode: 'adapter' (LoRA only), 'merged' (base+LoRA), or 'full' (everything)
            tag: Optional tag for this save
            push_to_hub: Whether to push to HuggingFace Hub
            hub_model_id: HuggingFace Hub repository ID

        Returns:
            Dict with save paths and URLs
        """
        try:
            self._update_activity()

            if self.model is None:
                raise RuntimeError("Model not initialized. Call initialize() first.")

            logger.info("SAVING STATE")

            logger.info(f"Step: {self.current_step}")
            logger.info(f"Mode: {mode}")

            # Save checkpoint
            if tag is None:
                tag = f"step_{self.current_step}"

            save_path = self._save_checkpoint_internal(tag=tag)

            # Upload to S3/R2 if configured
            result = {
                "status": "success",
                "step": self.current_step,
                "tag": tag,
                "local_path": save_path,
                "mode": mode,
            }

            try:
                from modal_runtime.s3_client import (
                    upload_directory,
                    generate_signed_url,
                )
                from datetime import datetime, timezone, timedelta

                logger.info("\nUploading checkpoint to S3/R2...")
                upload_result = upload_directory(
                    local_path=save_path,
                    s3_prefix=f"tenants/{self.user_id}/runs/{self.run_id}/checkpoints/{tag}/",
                )

                s3_uri = upload_result["s3_uri"]
                result["s3_uri"] = s3_uri

                # Generate signed download URL (valid for 1 hour)
                download_url = generate_signed_url(s3_uri, expiration=3600)
                download_expires_at = (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat()

                result["download_url"] = download_url
                result["download_expires_at"] = download_expires_at

                logger.info(
                    f"✓ Uploaded {upload_result.get('files_uploaded', 0)} files to S3/R2"
                )

            except Exception as e:
                logger.info(f"Warning: Failed to upload to S3/R2: {e}")
                result["s3_upload_error"] = str(e)

            # Push to HuggingFace Hub if requested
            if push_to_hub and hub_model_id:
                try:
                    import os

                    hf_token = os.environ.get("HF_TOKEN")

                    if not hf_token:
                        logger.info("Warning: HF_TOKEN not set, skipping Hub push")
                        result["hub_push_error"] = "HF_TOKEN not configured"
                    else:
                        logger.info(f"\nPushing to HuggingFace Hub: {hub_model_id}...")

                        model_to_save = self.accelerator.unwrap_model(self.model)

                        # Push model to Hub
                        model_to_save.push_to_hub(
                            hub_model_id,
                            token=hf_token,
                            commit_message=f"Checkpoint at step {self.current_step}",
                            private=True,
                        )

                        # Also push tokenizer
                        self.tokenizer.push_to_hub(
                            hub_model_id,
                            token=hf_token,
                            commit_message=f"Tokenizer at step {self.current_step}",
                            private=True,
                        )

                        result["pushed_to_hub"] = True
                        result["hub_model_id"] = hub_model_id
                        logger.info(f"✓ Pushed to Hub: {hub_model_id}")

                except Exception as e:
                    logger.info(f"Warning: Failed to push to Hub: {e}")
                    result["hub_push_error"] = str(e)
                    result["pushed_to_hub"] = False
            else:
                result["pushed_to_hub"] = False

            # Commit volume
            data_volume.commit()

            logger.info("✓ STATE SAVED")

            return result

        except Exception as e:
            error_msg = f"Save state failed: {str(e)}\n{traceback.format_exc()}"
            logger.info(f"\n❌ ERROR: {error_msg}")
            raise

    @modal.method()
    def get_state(self) -> Dict[str, Any]:
        """Get current session state.

        Returns:
            Dict with session information
        """
        gpu_summary = get_gpu_summary() if torch.cuda.is_available() else {}

        return {
            "status": "active" if self.model is not None else "uninitialized",
            "user_id": self.user_id,
            "run_id": self.run_id,
            "current_step": self.current_step,
            "last_checkpoint_step": self.last_checkpoint_step,
            "accumulation_count": self.accumulation_count,
            "accumulation_steps": self.accumulation_steps,
            "last_activity": time.time() - self.last_activity_time
            if self.last_activity_time
            else None,
            "num_processes": self.accelerator.num_processes if self.accelerator else 0,
            "config": self.config,
            "gpu_summary": gpu_summary,
        }

    @modal.method()
    def tokenize(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text(s) using the model's tokenizer."""
        self._update_activity()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call initialize() first.")

        if not isinstance(texts, list):
            texts = [texts]

        encoded = [
            self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in encoded]

        return {"token_ids": encoded, "tokens": tokens}

    @modal.method()
    def detokenize(self, token_ids) -> Dict[str, Any]:
        """Detokenize token IDs to text."""
        self._update_activity()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call initialize() first.")

        if isinstance(token_ids[0], list):
            texts = [self.tokenizer.decode(ids) for ids in token_ids]
        else:
            texts = self.tokenizer.decode(token_ids)
        return {"text": texts}

    @modal.method()
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer configuration."""
        self._update_activity()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call initialize() first.")

        return {
            "vocab_size": len(self.tokenizer),
            "model_max_length": self.tokenizer.model_max_length,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
            "special_tokens": {
                "bos_token": self.tokenizer.bos_token,
                "eos_token": self.tokenizer.eos_token,
                "pad_token": self.tokenizer.pad_token,
                "unk_token": self.tokenizer.unk_token,
            },
        }

    @modal.method()
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        self._update_activity()

        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        model = self.accelerator.unwrap_model(self.model)
        config = model.config

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "base_model": self.config.get("base_model"),
            "architecture": config.model_type,
            "num_parameters": total_params,
            "num_trainable_parameters": trainable_params,
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "chat_template": self.tokenizer.chat_template
            if hasattr(self.tokenizer, "chat_template")
            else None,
        }

    @modal.method()
    def apply_chat_template(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = False
    ) -> Dict[str, Any]:
        """Apply chat template to messages."""
        self._update_activity()

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call initialize() first.")

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not have chat template support")

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        token_ids = self.tokenizer.encode(text)

        return {"text": text, "token_ids": token_ids}

    @modal.method()
    def generate_embeddings(
        self, texts: List[str], layer: int = -1, pooling: str = "mean"
    ) -> Dict[str, Any]:
        """Generate embeddings for texts."""
        self._update_activity()

        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        model = self.accelerator.unwrap_model(self.model)
        model.eval()

        embeddings_list = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(
                    self.accelerator.device
                )
                outputs = model(**inputs, output_hidden_states=True)

                # Get hidden states from specified layer
                hidden_states = outputs.hidden_states[layer]

                # Apply pooling
                if pooling == "mean":
                    embedding = hidden_states.mean(dim=1).squeeze()
                elif pooling == "last_token":
                    embedding = hidden_states[:, -1, :].squeeze()
                elif pooling == "cls_token":
                    embedding = hidden_states[:, 0, :].squeeze()

                embeddings_list.append(embedding.cpu().tolist())

        model.train()
        return {
            "embeddings": embeddings_list,
            "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
        }

    @modal.method()
    def train_grpo_step(
        self,
        prompts: List[str],
        reward_funcs: List[Any],
        num_generations: int = 8,
        beta: float = 0.0,
        loss_type: str = "grpo",
        max_prompt_length: int = 1024,
        max_completion_length: int = 2048,
    ) -> Dict[str, Any]:
        """Train one GRPO step.

        GRPO (Group Relative Policy Optimization) is a policy optimization method
        that doesn't require a reference model. It uses group-based rewards.

        Args:
            prompts: List of prompts to generate completions for
            reward_funcs: List of reward functions that take completions and return scores
            num_generations: Number of samples per prompt
            beta: KL penalty coefficient (0.0 = no KL penalty)
            loss_type: GRPO variant ("grpo", "dapo", "dr_grpo")
            max_prompt_length: Maximum prompt length
            max_completion_length: Maximum completion length

        Returns:
            Training metrics

        Example:
            >>> def length_reward(completions):
            ...     return [len(c) * 0.1 for c in completions]
            >>>
            >>> result = session.train_grpo_step(
            ...     prompts=["What is AI?"],
            ...     reward_funcs=[length_reward],
            ...     num_generations=8,
            ... )
        """
        self._update_activity()

        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        from modal_runtime.trl_trainers import create_grpo_trainer
        from datasets import Dataset

        # Create dataset from prompts
        dataset = Dataset.from_dict({"prompt": prompts})

        # Unwrap model for TRL
        model = self.accelerator.unwrap_model(self.model)

        # Create GRPO trainer
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=self.tokenizer,
            reward_funcs=reward_funcs,
            train_dataset=dataset,
            num_generations=num_generations,
            beta=beta,
            loss_type=loss_type,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=self.config.get("accumulation_steps", 1),
        )

        # Train one step
        # Note: TRL trainers use .train() for full training
        # For single step, we'd need to manually call trainer.step()
        # or use a single epoch with one batch
        trainer.train()

        self.current_step += 1

        return {
            "status": "success",
            "step": self.current_step,
            "message": "GRPO step completed",
        }

    # Internal helper methods

    def _save_checkpoint_internal(self, tag: Optional[str] = None) -> str:
        """Internal checkpoint save method."""
        if self.model is None:
            return ""

        paths = get_run_paths(self.user_id, self.run_id)
        if tag is None:
            tag = f"step_{self.current_step}"
        checkpoint_path = paths["lora_adapters"] / tag
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Unwrap model (Accelerate wraps it)
        model_to_save = self.accelerator.unwrap_model(self.model)

        # Save LoRA checkpoint (PEFT handles this)
        model_to_save.save_pretrained(checkpoint_path)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_path)

        # Save optimizer state
        opt_path = paths["optimizer_state"]
        torch.save(self.optimizer.state_dict(), opt_path)

        self.last_checkpoint_step = self.current_step
        logger.info(f"✓ Checkpoint saved at step {self.current_step}")

        return str(checkpoint_path)

    def _update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_time = time.time()

    def _background_monitor(self):
        """Background thread for auto-checkpoint monitoring."""
        logger.info("Background monitor thread started")

        while self.should_monitor:
            time.sleep(60)  # Check every minute

            if not self.should_monitor:
                break

            if self.model is None:
                continue

            # Check if checkpoint needed
            steps_since_checkpoint = self.current_step - self.last_checkpoint_step
            if steps_since_checkpoint >= self.auto_checkpoint_interval:
                try:
                    logger.info("\n[Background] Auto-checkpoint triggered")
                    self._save_checkpoint_internal()
                    data_volume.commit()
                except Exception as e:
                    logger.info(f"[Background] Auto-checkpoint failed: {e}")

                logger.info("Background monitor thread stopped")
