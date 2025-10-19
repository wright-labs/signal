"""Multi-GPU training sessions with different GPU configurations.

This module defines multiple TrainingSession classes with different GPU configs.
Modal requires GPU config at class definition time, so we define each explicitly.
"""
import torch
import modal
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import threading
import traceback
from functools import wraps

# TODO: dig through this one significantly more


def log_errors(func):
    """Decorator to log errors and re-raise them with context."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"{func.__name__} failed: {str(e)}\n{traceback.format_exc()}"
            print(f"\n❌ ERROR: {error_msg}")
            raise
    return wrapper

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
    setup_optimizer,
    tokenize_batch,
    save_lora_checkpoint,
    save_optimizer_state,
    load_optimizer_state,
    get_run_paths,
    save_run_config,
    load_run_config,
    find_latest_checkpoint,
    compute_forward_backward,
)
from modal_runtime.gpu_monitor import (
    get_gpu_stats,
    get_gpu_summary,
    setup_multi_gpu_model,
    print_gpu_stats,
)


# Base class implementation that will be reused
class TrainingSessionBase:
    """Base training session implementation (shared by all GPU configs)."""
    
    # Instance variables
    model: Any = None
    tokenizer: Any = None
    optimizer: Any = None
    scheduler: Any = None
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
    num_gpus: int = 0
    is_multi_gpu: bool = False
    gpu_config_str: str = ""
    
    def container_startup_impl(self, gpu_config: str):
        """Shared container startup logic."""
        self.gpu_config_str = gpu_config
        
        print("=" * 80)
        print(f"CONTAINER STARTED - GPU Config: {gpu_config}")
        print("=" * 80)
        
        # Detect GPUs
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.is_multi_gpu = self.num_gpus > 1
            print(f"✓ Detected {self.num_gpus} GPU(s)")
            
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️  No CUDA GPUs detected!")
        
        # Start background monitoring
        self.should_monitor = True
        self.monitor_thread = threading.Thread(
            target=self._background_monitor,
            daemon=True
        )
        self.monitor_thread.start()
        print("✓ Background monitoring thread started")
        print("=" * 80)
    
    def container_shutdown_impl(self):
        """Shared container shutdown logic."""
        print("=" * 80)
        print("CONTAINER SHUTTING DOWN")
        print("=" * 80)
        
        # Stop monitoring
        self.should_monitor = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Auto-save
        if self.model is not None:
            try:
                print(f"Auto-saving checkpoint at step {self.current_step}...")
                self._save_checkpoint_internal(tag=f"autosave-{int(time.time())}")
                data_volume.commit()
                print("✓ Auto-save complete")
            except Exception as e:
                print(f"⚠️  Auto-save failed: {e}")
        
        # Clean up
        if self.model is not None:
            del self.model
        if self.optimizer is not None:
            del self.optimizer
        
        torch.cuda.empty_cache()
        print("✓ GPU memory cleaned up")
        print("✓ Container shutdown complete")
    
    @log_errors
    def initialize_impl(
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
        accumulation_steps: int = 1,
        auto_checkpoint_interval: int = 100,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Initialize training session with model and optimizer."""
        self._update_activity()
        
        print("\n" + "=" * 80)
        print(f"INITIALIZING TRAINING SESSION")
        print(f"Run ID: {run_id}")
        print(f"Model: {base_model}")
        print(f"GPU Config: {self.gpu_config_str} ({self.num_gpus} GPUs)")
        print("=" * 80)
        
        # Store config
        self.user_id = user_id
        self.run_id = run_id
        self.accumulation_steps = accumulation_steps
        self.auto_checkpoint_interval = auto_checkpoint_interval
        
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
            "gpu_config": self.gpu_config_str,
            "num_gpus": self.num_gpus,
        }
        
        # Get run paths
        run_paths = get_run_paths(user_id, run_id)
        save_run_config(run_paths["config_path"], self.config)
        
        # Load model and tokenizer
        print(f"\n1. Loading model: {base_model}")
        print(f"   Quantization: {'8bit' if load_in_8bit else '4bit' if load_in_4bit else 'none'}")
        
        # For multi-GPU, load on cuda:0 only (DataParallel will distribute)
        device_map = "cuda:0" if self.is_multi_gpu else "auto"
        
        # Disable quantization for multi-GPU
        if self.is_multi_gpu:
            load_in_4bit = False
            load_in_8bit = False
            print("   Note: Quantization disabled for multi-GPU")
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            base_model,
            max_seq_length=max_seq_length,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        
        print("✓ Model and tokenizer loaded")
        
        # Apply LoRA
        print(f"\n2. Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        self.model = apply_lora_to_model(
            self.model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✓ LoRA applied")
        print(f"   Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Multi-GPU setup
        if self.is_multi_gpu:
            print(f"\n3. Setting up multi-GPU training ({self.num_gpus} GPUs)")
            self.model = setup_multi_gpu_model(self.model, strategy="data_parallel")
            print(f"✓ Model distributed across {self.num_gpus} GPUs")
        else:
            print(f"\n3. Single GPU mode")
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            print(f"\n4. Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint_path = Path(run_paths["adapters_dir"]) / resume_from_checkpoint
            self.model = load_lora_checkpoint(self.model, str(checkpoint_path))
            
            # Extract step from checkpoint name
            if "step_" in resume_from_checkpoint:
                try:
                    step_str = resume_from_checkpoint.split("step_")[1]
                    self.current_step = int(step_str)
                    self.last_checkpoint_step = self.current_step
                    print(f"   Resumed at step {self.current_step}")
                except:
                    pass
        
        # Setup optimizer
        step_num = 4 if resume_from_checkpoint else 4
        print(f"\n{step_num}. Setting up optimizer: {optimizer}")
        self.optimizer = setup_optimizer(
            self.model,
            optimizer_type=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Load optimizer state if resuming
        if resume_from_checkpoint:
            opt_state_path = Path(run_paths["adapters_dir"]) / resume_from_checkpoint / "optimizer.pt"
            if opt_state_path.exists():
                self.optimizer = load_optimizer_state(self.optimizer, str(opt_state_path))
                print("   ✓ Optimizer state loaded")
        
        print("✓ Optimizer ready")
        
        print("\n" + "=" * 80)
        print("✓ INITIALIZATION COMPLETE")
        print(f"   Model: {base_model}")
        print(f"   GPUs: {self.num_gpus}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Current step: {self.current_step}")
        print("=" * 80)
        
        return {
            "status": "success",
            "run_id": run_id,
            "base_model": base_model,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "current_step": self.current_step,
            "num_gpus": self.num_gpus,
            "is_multi_gpu": self.is_multi_gpu,
            "config": self.config,
        }
    
    @log_errors
    def forward_backward_impl(
        self,
        batch_data: List[Dict[str, Any]],
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute forward pass and backward gradients."""
        self._update_activity()
        
        if loss_kwargs is None:
            loss_kwargs = {}
        
        # Tokenize
        inputs = tokenize_batch(
            batch_data,
            self.tokenizer,
            max_length=self.config["max_seq_length"],
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Set model to training mode
        self.model.train()
        
        # FORWARD PASS (explicit and clean)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels", inputs["input_ids"]),
        )
        
        # COMPUTE LOSS (separate step)
        from modal_runtime.loss_functions import compute_loss_from_outputs
        loss, loss_metrics = compute_loss_from_outputs(
            outputs,
            inputs.get("labels"),
            loss_fn,
            **loss_kwargs
        )
        
        # BACKWARD PASS
        loss.backward()
        
        # Compute gradient statistics
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        
        grad_stats = {
            "grad_norm": grad_norm.item(),
            **loss_metrics,
        }
        
        # Track accumulation
        self.accumulation_count += 1
        
        return {
            "status": "success",
            "loss": loss.item(),
            "step": self.current_step,
            "accumulation_count": self.accumulation_count,
            "grad_norm": grad_stats.get("grad_norm", 0.0),
            "metrics": grad_stats,
            "num_gpus": self.num_gpus,
        }
    
    @log_errors
    def optim_step_impl(
        self,
        learning_rate: Optional[float] = None,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply optimizer update."""
        self._update_activity()
        
        # Only step if accumulation complete
        if self.accumulation_count < self.accumulation_steps:
            return {
                "status": "accumulating",
                "step": self.current_step,
                "accumulation_count": self.accumulation_count,
            }
        
        # Override LR if provided
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Reset accumulation
        self.accumulation_count = 0
        self.current_step += 1
        
        # Auto-checkpoint
        if (self.auto_checkpoint_interval > 0 and 
            self.current_step % self.auto_checkpoint_interval == 0):
            print(f"\n🔄 Auto-checkpoint at step {self.current_step}")
            self._save_checkpoint_internal(tag=f"step_{self.current_step}")
            self.last_checkpoint_step = self.current_step
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "status": "success",
            "step": self.current_step,
            "learning_rate": current_lr,
            "num_gpus": self.num_gpus,
        }
    
    @log_errors
    def sample_impl(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        return_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Generate text samples."""
        self._update_activity()
        
        # Use base model for generation (unwrap if DataParallel)
        model_to_use = self.model.module if self.is_multi_gpu else self.model
        
        outputs = []
        token_ids_list = []
        
        model_to_use.eval()
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model_to_use.device)
            
            with torch.no_grad():
                generated = model_to_use.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                )
            
            output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(output_text)
            token_ids_list.append(generated[0].tolist())
        
        model_to_use.train()
        
        return {
            "status": "success",
            "outputs": outputs,
            "token_ids": token_ids_list,
            "logprobs": None,
            "step": self.current_step,
        }
    
    @log_errors
    def save_state_impl(
        self,
        mode: str = "adapter",
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model checkpoint."""
        self._update_activity()
        
        if tag is None:
            tag = f"step_{self.current_step}"
        
        save_path = self._save_checkpoint_internal(tag=tag)
        data_volume.commit()
        
        return {
            "status": "success",
            "save_path": save_path,
            "step": self.current_step,
            "mode": mode,
        }
    
    def get_state_impl(self) -> Dict[str, Any]:
        """Get current session state."""
        gpu_summary = get_gpu_summary()
        
        return {
            "status": "active" if self.model is not None else "uninitialized",
            "run_id": self.run_id,
            "current_step": self.current_step,
            "last_checkpoint_step": self.last_checkpoint_step,
            "accumulation_count": self.accumulation_count,
            "accumulation_steps": self.accumulation_steps,
            "gpu_config": self.gpu_config_str,
            "num_gpus": self.num_gpus,
            "is_multi_gpu": self.is_multi_gpu,
            "gpu_summary": gpu_summary,
        }
    
    def _save_checkpoint_internal(self, tag: str) -> str:
        """Internal checkpoint save."""
        run_paths = get_run_paths(self.user_id, self.run_id)
        save_dir = Path(run_paths["adapters_dir"]) / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters (unwrap if multi-GPU)
        model_to_save = self.model.module if self.is_multi_gpu else self.model
        save_lora_checkpoint(model_to_save, str(save_dir))
        
        # Save optimizer
        save_optimizer_state(self.optimizer, str(save_dir / "optimizer.pt"))
        
        return str(save_dir)
    
    def _update_activity(self):
        """Update last activity time."""
        self.last_activity_time = time.time()
    
    def _background_monitor(self):
        """Background thread for auto-checkpoint."""
        while self.should_monitor:
            time.sleep(60)  # Check every minute


# Now define each GPU configuration as a separate class at module level
# This avoids Modal's LocalFunctionError

@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="L40S:1",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_L40S_1(TrainingSessionBase):
    """TrainingSession with 1x L40S GPU."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("L40S:1")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="L40S:2",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_L40S_2(TrainingSessionBase):
    """TrainingSession with 2x L40S GPUs."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("L40S:2")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="L40S:4",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_L40S_4(TrainingSessionBase):
    """TrainingSession with 4x L40S GPUs."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("L40S:4")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="A100-80GB:1",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_A100_80GB_1(TrainingSessionBase):
    """TrainingSession with 1x A100-80GB GPU."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("A100-80GB:1")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="A100-80GB:2",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_A100_80GB_2(TrainingSessionBase):
    """TrainingSession with 2x A100-80GB GPUs."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("A100-80GB:2")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="A100-80GB:4",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_A100_80GB_4(TrainingSessionBase):
    """TrainingSession with 4x A100-80GB GPUs."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("A100-80GB:4")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="A100-80GB:8",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_A100_80GB_8(TrainingSessionBase):
    """TrainingSession with 8x A100-80GB GPUs."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("A100-80GB:8")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="H100:1",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_H100_1(TrainingSessionBase):
    """TrainingSession with 1x H100 GPU."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("H100:1")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()


@app.cls(
    image=TRAINING_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, s3_secret],
    gpu="H100:4",
    timeout=2 * HOURS,
    container_idle_timeout=20 * 60,
    allow_concurrent_inputs=10,
)
class TrainingSession_H100_4(TrainingSessionBase):
    """TrainingSession with 4x H100 GPUs."""
    
    @modal.enter()
    def container_startup(self):
        self.container_startup_impl("H100:4")
    
    @modal.exit()
    def container_shutdown(self):
        self.container_shutdown_impl()
    
    @modal.method()
    def initialize(self, **kwargs):
        return self.initialize_impl(**kwargs)
    
    @modal.method()
    def forward_backward(self, **kwargs):
        return self.forward_backward_impl(**kwargs)
    
    @modal.method()
    def optim_step(self, **kwargs):
        return self.optim_step_impl(**kwargs)
    
    @modal.method()
    def sample(self, **kwargs):
        return self.sample_impl(**kwargs)
    
    @modal.method()
    def save_state(self, **kwargs):
        return self.save_state_impl(**kwargs)
    
    @modal.method()
    def get_state(self):
        return self.get_state_impl()
