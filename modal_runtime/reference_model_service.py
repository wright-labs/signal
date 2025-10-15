"""Separate Modal container for reference model inference.

This service provides a dedicated container for loading large reference models
that don't fit in memory alongside the training model.
"""

import modal
import torch
from typing import List, Dict, Any, Optional

from modal_runtime.app import (
    app,
    TRAINING_IMAGE,
    huggingface_secret,
)


@app.cls(
    image=TRAINING_IMAGE,
    secrets=[huggingface_secret],
    gpu="A100-80GB:1",  # Large GPU for reference models
    timeout=600,  # 10 minute timeout
    container_idle_timeout=20 * 60,  # 20 minutes idle
    allow_concurrent_inputs=20,  # High concurrency for inference
)
class ReferenceModelService:
    """Dedicated service for reference model inference.
    
    This service:
    - Loads large reference models that don't fit in training container
    - Provides batched inference for efficiency
    - Supports multiple models (lazy loading)
    - Optimized for inference (no training)
    """
    
    # Instance variables
    models: Dict[str, Any] = {}
    tokenizers: Dict[str, Any] = {}
    current_model: Optional[str] = None
    
    @modal.enter()
    def load_initial_model(self):
        """Called when container starts."""
        print("=" * 80)
        print("REFERENCE MODEL SERVICE STARTED")
        print("=" * 80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️ No CUDA GPU detected!")
        
        print("Ready to load reference models on demand")
    
    @modal.exit()
    def cleanup(self):
        """Called when container shuts down."""
        print("Reference model service shutting down...")
        
        # Clean up models
        for model_name in list(self.models.keys()):
            del self.models[model_name]
            del self.tokenizers[model_name]
        
        self.models.clear()
        self.tokenizers.clear()
        torch.cuda.empty_cache()
    
    def _load_model_if_needed(
        self,
        model_name: str,
        load_in_8bit: bool = True,
    ) -> None:
        """Load model if not already loaded.
        
        Args:
            model_name: HuggingFace model name
            load_in_8bit: Whether to load in 8-bit
        """
        if model_name in self.models:
            self.current_model = model_name
            return
        
        print(f"\nLoading reference model: {model_name}")
        print(f"Quantization: {'8-bit' if load_in_8bit else 'none'}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        
        # Disable gradients (reference model is frozen)
        for param in model.parameters():
            param.requires_grad = False
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Store in cache
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        self.current_model = model_name
        
        print(f"✓ Reference model loaded: {model_name}")
    
    @modal.method()
    def compute_log_probs(
        self,
        model_name: str,
        input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]] = None,
        load_in_8bit: bool = True,
    ) -> Dict[str, Any]:
        """Compute log probabilities for given inputs.
        
        Args:
            model_name: HuggingFace model name
            input_ids: List of token ID sequences [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            load_in_8bit: Whether to load model in 8-bit
            
        Returns:
            Dictionary with log_probs and other info
        """
        # Load model if needed
        self._load_model_if_needed(model_name, load_in_8bit=load_in_8bit)
        
        model = self.models[model_name]
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, device=model.device)
        if attention_mask is not None:
            attention_mask_tensor = torch.tensor(attention_mask, device=model.device)
        else:
            attention_mask_tensor = None
        
        # Compute log probs
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                labels=input_ids_tensor,  # For computing loss
            )
            
            # Negative loss = log prob
            log_probs = -outputs.loss
            
            # Handle per-token log probs if needed
            if log_probs.dim() == 0:
                log_probs = log_probs.unsqueeze(0).expand(input_ids_tensor.shape[0])
        
        return {
            "log_probs": log_probs.cpu().tolist(),
            "model_name": model_name,
            "batch_size": len(input_ids),
        }
    
    @modal.method()
    def compute_log_probs_batch(
        self,
        model_name: str,
        batches: List[Dict[str, Any]],
        load_in_8bit: bool = True,
    ) -> List[Dict[str, Any]]:
        """Compute log probabilities for multiple batches.
        
        Args:
            model_name: HuggingFace model name
            batches: List of batch dicts with 'input_ids' and optional 'attention_mask'
            load_in_8bit: Whether to load model in 8-bit
            
        Returns:
            List of result dictionaries
        """
        # Load model once
        self._load_model_if_needed(model_name, load_in_8bit=load_in_8bit)
        
        results = []
        for batch in batches:
            result = self.compute_log_probs(
                model_name=model_name,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                load_in_8bit=load_in_8bit,
            )
            results.append(result)
        
        return results
    
    @modal.method()
    def compute_rewards(
        self,
        model_name: str,
        input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]] = None,
        load_in_8bit: bool = True,
    ) -> Dict[str, Any]:
        """Compute reward scores using a reward model.
        
        Args:
            model_name: HuggingFace reward model name
            input_ids: List of token ID sequences [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            load_in_8bit: Whether to load model in 8-bit
            
        Returns:
            Dictionary with reward scores
        """
        # Load model if needed
        self._load_model_if_needed(model_name, load_in_8bit=load_in_8bit)
        
        model = self.models[model_name]
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, device=model.device)
        if attention_mask is not None:
            attention_mask_tensor = torch.tensor(attention_mask, device=model.device)
        else:
            attention_mask_tensor = None
        
        # Compute rewards
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )
            
            # Use last token logits mean as reward score
            # (this is a common pattern for reward models)
            rewards = outputs.logits[:, -1, :].mean(dim=-1)
        
        return {
            "rewards": rewards.cpu().tolist(),
            "model_name": model_name,
            "batch_size": len(input_ids),
        }
    
    @modal.method()
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    @modal.method()
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from memory.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            True if model was unloaded, False if not loaded
        """
        if model_name not in self.models:
            return False
        
        print(f"Unloading reference model: {model_name}")
        
        del self.models[model_name]
        del self.tokenizers[model_name]
        
        if self.current_model == model_name:
            self.current_model = None
        
        torch.cuda.empty_cache()
        
        print(f"✓ Reference model unloaded: {model_name}")
        return True
    
    @modal.method()
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics.
        
        Returns:
            Dictionary with memory info
        """
        if not torch.cuda.is_available():
            return {"error": "No CUDA available"}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "num_loaded_models": len(self.models),
            "loaded_models": list(self.models.keys()),
        }

