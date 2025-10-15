"""Reference model cache with LRU eviction and memory management.

This module provides a hybrid caching strategy for reference models:
- In-memory LRU cache for frequently used models
- Automatic quantization to save GPU memory
- Memory-aware eviction policies
"""

import torch
from collections import OrderedDict
from typing import Any, Optional, Dict
import threading
from pathlib import Path


class ReferenceModelCache:
    """LRU cache for reference models with memory management.
    
    This cache:
    - Keeps up to max_models in memory
    - Uses 8-bit quantization by default to save memory
    - Evicts least-recently-used models when full
    - Thread-safe for concurrent access
    
    Example:
        cache = ReferenceModelCache(max_models=2)
        ref_model = cache.get_or_load("meta-llama/Llama-3.2-1B", quantize=True)
    """
    
    def __init__(
        self,
        max_models: int = 2,
        quantize_by_default: bool = True,
        device: str = "cuda",
    ):
        """Initialize reference model cache.
        
        Args:
            max_models: Maximum number of models to keep in memory
            quantize_by_default: Whether to quantize models by default
            device: Device to load models on
        """
        self.max_models = max_models
        self.quantize_by_default = quantize_by_default
        self.device = device
        
        # LRU cache: model_name -> model
        self.cache: OrderedDict[str, Any] = OrderedDict()
        
        # Model metadata: model_name -> metadata dict
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def get_or_load(
        self,
        model_name: str,
        quantize: Optional[bool] = None,
        load_in_8bit: Optional[bool] = None,
        load_in_4bit: bool = False,
    ) -> Any:
        """Get cached model or load if not present.
        
        Args:
            model_name: HuggingFace model name
            quantize: Whether to quantize (overrides default)
            load_in_8bit: Whether to load in 8-bit (deprecated, use quantize)
            load_in_4bit: Whether to load in 4-bit
            
        Returns:
            Loaded model
        """
        with self._lock:
            # Check if model is in cache
            if model_name in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(model_name)
                print(f"✓ Reference model cache hit: {model_name}")
                return self.cache[model_name]
            
            # Model not in cache, load it
            print(f"Loading reference model: {model_name}")
            
            # Determine quantization
            if quantize is None:
                quantize = self.quantize_by_default
            if load_in_8bit is not None:
                quantize = load_in_8bit
            
            # Load model
            model = self._load_model(
                model_name,
                quantize=quantize,
                load_in_4bit=load_in_4bit,
            )
            
            # Evict oldest model if cache full
            if len(self.cache) >= self.max_models:
                self._evict_oldest()
            
            # Add to cache
            self.cache[model_name] = model
            self.metadata[model_name] = {
                "quantized": quantize or load_in_4bit,
                "device": str(self.device),
            }
            
            print(f"✓ Reference model loaded and cached: {model_name}")
            return model
    
    def _load_model(
        self,
        model_name: str,
        quantize: bool = True,
        load_in_4bit: bool = False,
    ) -> Any:
        """Load a model from HuggingFace.
        
        Args:
            model_name: HuggingFace model name
            quantize: Whether to load in 8-bit
            load_in_4bit: Whether to load in 4-bit
            
        Returns:
            Loaded model
        """
        from transformers import AutoModelForCausalLM
        
        print(f"  Loading from HuggingFace: {model_name}")
        print(f"  Quantization: {'4-bit' if load_in_4bit else '8-bit' if quantize else 'none'}")
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=quantize and not load_in_4bit,
            load_in_4bit=load_in_4bit,
            device_map=self.device if not (quantize or load_in_4bit) else "auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Set to eval mode (reference models are frozen)
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _evict_oldest(self) -> None:
        """Evict the least-recently-used model from cache."""
        if len(self.cache) == 0:
            return
        
        # Get oldest model (first in OrderedDict)
        oldest_model_name = next(iter(self.cache))
        oldest_model = self.cache[oldest_model_name]
        
        print(f"Evicting reference model from cache: {oldest_model_name}")
        
        # Remove from cache
        del self.cache[oldest_model_name]
        if oldest_model_name in self.metadata:
            del self.metadata[oldest_model_name]
        
        # Delete model and clear GPU memory
        del oldest_model
        torch.cuda.empty_cache()
        
        print(f"✓ Reference model evicted: {oldest_model_name}")
    
    def clear(self) -> None:
        """Clear all models from cache."""
        with self._lock:
            print(f"Clearing reference model cache ({len(self.cache)} models)")
            
            for model_name, model in self.cache.items():
                del model
            
            self.cache.clear()
            self.metadata.clear()
            torch.cuda.empty_cache()
            
            print("✓ Reference model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache information
        """
        with self._lock:
            return {
                "num_models": len(self.cache),
                "max_models": self.max_models,
                "models": list(self.cache.keys()),
                "metadata": self.metadata.copy(),
            }
    
    def contains(self, model_name: str) -> bool:
        """Check if model is in cache.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            True if model is in cache
        """
        with self._lock:
            return model_name in self.cache
    
    def remove(self, model_name: str) -> bool:
        """Remove a specific model from cache.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            True if model was removed, False if not in cache
        """
        with self._lock:
            if model_name not in self.cache:
                return False
            
            model = self.cache[model_name]
            del self.cache[model_name]
            if model_name in self.metadata:
                del self.metadata[model_name]
            
            del model
            torch.cuda.empty_cache()
            
            print(f"✓ Reference model removed from cache: {model_name}")
            return True
    
    def __len__(self) -> int:
        """Get number of models in cache."""
        with self._lock:
            return len(self.cache)
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model is in cache (supports 'in' operator)."""
        return self.contains(model_name)
    
    def __repr__(self) -> str:
        with self._lock:
            return f"<ReferenceModelCache(size={len(self.cache)}/{self.max_models}, models={list(self.cache.keys())})>"


# Global cache instance (singleton pattern)
_global_cache: Optional[ReferenceModelCache] = None
_cache_lock = threading.Lock()


def get_global_reference_cache(
    max_models: int = 2,
    quantize_by_default: bool = True,
) -> ReferenceModelCache:
    """Get the global reference model cache instance.
    
    This implements a singleton pattern for the cache.
    
    Args:
        max_models: Maximum number of models to keep in memory
        quantize_by_default: Whether to quantize models by default
        
    Returns:
        Global ReferenceModelCache instance
    """
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = ReferenceModelCache(
                max_models=max_models,
                quantize_by_default=quantize_by_default,
            )
        
        return _global_cache


def clear_global_reference_cache() -> None:
    """Clear the global reference model cache."""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
            _global_cache = None

