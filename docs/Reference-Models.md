# Reference Models Guide

Reference models are crucial for RL training to prevent policy collapse by computing KL divergence penalties. Signal provides efficient reference model management with hybrid caching strategies.

## Overview

Reference models serve two main purposes:

1. **KL Divergence Penalty**: Prevent the policy from diverging too far from the reference
2. **Reward Model Evaluation**: Compare policy outputs against a baseline

Signal's reference model infrastructure features:
- **LRU cache** for frequently used models
- **Automatic quantization** to save GPU memory
- **Hybrid loading** strategy (in-memory or separate service)
- **Thread-safe** access for concurrent requests

## Basic Usage

### Specifying a Reference Model

```python
result = await training_client.forward_backward(
    batch_data=ppo_batch,
    loss_fn="enhanced_ppo",
    loss_kwargs={
        "reference_model": "meta-llama/Llama-3.2-1B",  # HuggingFace model name
        "beta": 0.01,  # KL penalty coefficient
    }
)
```

The reference model is automatically:
1. Loaded from HuggingFace
2. Quantized to 8-bit (by default)
3. Cached for reuse
4. Frozen (no gradients)

### KL Divergence

The KL penalty is computed as:

```
KL(π_θ || π_ref) = E[log π_θ(a|s) - log π_ref(a|s)]
Loss = policy_loss + beta * KL
```

## Reference Model Cache

### Architecture

Signal uses an **LRU (Least Recently Used) cache**:

```
┌─────────────────────────────────────┐
│   Reference Model Cache (LRU)       │
├─────────────────────────────────────┤
│ Model 1 (most recent)               │
│ Model 2                             │
│ ... (up to max_models)              │
└─────────────────────────────────────┘
         │
         ├─ Cache Hit → Use cached model
         └─ Cache Miss → Load and evict oldest
```

### Configuration

```python
from modal_runtime.reference_model_cache import get_global_reference_cache

cache = get_global_reference_cache(
    max_models=2,  # Max models in memory
    quantize_by_default=True,  # 8-bit quantization
)
```

### Cache Management

```python
# Check if model is cached
if cache.contains("meta-llama/Llama-3.2-1B"):
    print("Model in cache!")

# Load or get cached model
model = cache.get_or_load("meta-llama/Llama-3.2-1B")

# Get cache statistics
info = cache.get_cache_info()
print(f"Models in cache: {info['num_models']}/{info['max_models']}")
print(f"Cached models: {info['models']}")

# Remove a model from cache
cache.remove("meta-llama/Llama-3.2-1B")

# Clear entire cache
cache.clear()
```

## Memory Management

### Quantization Options

**8-bit quantization** (default):
```python
model = cache.get_or_load(
    "meta-llama/Llama-3.2-1B",
    quantize=True,  # Load in 8-bit
)
```

**4-bit quantization** (even more memory-efficient):
```python
model = cache.get_or_load(
    "meta-llama/Llama-3.2-1B",
    load_in_4bit=True,
)
```

**No quantization**:
```python
model = cache.get_or_load(
    "meta-llama/Llama-3.2-1B",
    quantize=False,
)
```

### Memory Savings

| Precision | Memory Usage | Inference Speed | Quality |
|-----------|--------------|-----------------|---------|
| Float32 | 100% | 100% | 100% |
| Float16 | 50% | 120% | 99.9% |
| 8-bit | 25% | 80% | 99% |
| 4-bit | 12.5% | 60% | 95% |

For reference models (frozen, no training):
- **8-bit is recommended** (good balance)
- **4-bit works well** for very large models
- **Float16/32 rarely needed** for reference models

### LRU Eviction

When cache is full:
1. Oldest (least recently used) model is evicted
2. GPU memory is freed
3. New model is loaded and cached

```
Cache (max 2):
Initial: [Model A, Model B]

Access Model C:
Step 1: Evict Model A (oldest)
Step 2: Load Model C
Result: [Model B, Model C]
```

## Separate Reference Model Service

For very large reference models that don't fit in the training container, Signal provides a **dedicated Modal service**:

### Architecture

```
┌──────────────────────────────────┐
│   Training Container             │
│   ├─ Policy Model (trainable)    │
│   └─ Small Ref Models (cached)   │
└──────────────────────────────────┘
            │
            │ (API call)
            ▼
┌──────────────────────────────────┐
│   Reference Model Service        │
│   (Large GPU, inference-only)    │
│   └─ Large Ref Models            │
└──────────────────────────────────┘
```

### Usage

The service is automatically used when:
1. Reference model is too large for training container
2. Explicit service call is made

```python
from modal_runtime.reference_model_service import ReferenceModelService

# Get service instance
service = ReferenceModelService()

# Compute log probs for KL divergence
result = await service.compute_log_probs.remote(
    model_name="meta-llama/Llama-3.2-70B",  # Large model
    input_ids=batch["input_ids"].tolist(),
    attention_mask=batch["attention_mask"].tolist(),
)

log_probs = torch.tensor(result["log_probs"])
```

### Service Configuration

The service runs on:
- **GPU**: A100-80GB (large memory)
- **Idle timeout**: 20 minutes
- **Concurrency**: 20 (high throughput)
- **Quantization**: 8-bit by default

## Best Practices

### 1. Choose Appropriate Reference Models

**Same architecture as policy model:**
```python
# Good: Same model family
policy: "meta-llama/Llama-3.2-1B-Instruct" (being trained)
reference: "meta-llama/Llama-3.2-1B" (base model)

# Avoid: Different architectures
policy: "meta-llama/Llama-3.2-1B"
reference: "mistralai/Mistral-7B-v0.1"  # Different architecture
```

### 2. Start with Base Models

Use the **base (pre-trained) model** as reference, not an instruct-tuned version:
```python
reference_model="meta-llama/Llama-3.2-1B"  # Base model ✓
# NOT "meta-llama/Llama-3.2-1B-Instruct"  # Instruct model ✗
```

### 3. Cache Reuse

Reuse the same reference model across training runs:
```python
# Run 1
await training_client1.forward_backward(..., loss_kwargs={
    "reference_model": "meta-llama/Llama-3.2-1B",  # Loads and caches
})

# Run 2 (cache hit!)
await training_client2.forward_backward(..., loss_kwargs={
    "reference_model": "meta-llama/Llama-3.2-1B",  # Uses cached
})
```

### 4. Monitor KL Divergence

Keep KL divergence in check:
```python
result = await training_client.forward_backward(...)

kl = result["metrics"]["kl_divergence"]
if kl > 0.1:
    print("Warning: High KL divergence! Consider:")
    print("- Increasing beta (KL penalty)")
    print("- Decreasing learning rate")
    print("- Using conservative_ppo")
```

### 5. Memory-Constrained Scenarios

If running out of GPU memory:

**Option 1: Use 8-bit or 4-bit quantization**
```python
result = await training_client.forward_backward(..., loss_kwargs={
    "reference_model": "meta-llama/Llama-3.2-1B",
    "beta": 0.01,
})
# Automatically uses 8-bit quantization
```

**Option 2: Reduce cache size**
```python
from modal_runtime.reference_model_cache import get_global_reference_cache

cache = get_global_reference_cache(max_models=1)  # Only keep 1 model
```

**Option 3: Use separate service**
```python
# Large reference models automatically routed to separate service
result = await training_client.forward_backward(..., loss_kwargs={
    "reference_model": "meta-llama/Llama-3.2-70B",  # Routed to service
})
```

## Troubleshooting

### Model Loading Errors

**Issue**: `OSError: Unable to load weights`

**Solutions**:
1. Check model name spelling
2. Verify HuggingFace access token (for gated models)
3. Check disk space
4. Try without quantization

```python
# Set HF token
import os
os.environ["HF_TOKEN"] = "your-token"

# Load without quantization
model = cache.get_or_load(
    "meta-llama/Llama-3.2-1B",
    quantize=False,
)
```

### Out of Memory

**Issue**: `CUDA out of memory`

**Solutions**:
1. Enable quantization
2. Reduce cache size
3. Use separate service for large models
4. Reduce batch size

```python
# Enable 4-bit quantization
model = cache.get_or_load(
    model_name,
    load_in_4bit=True,
)

# Clear cache before loading
cache.clear()
model = cache.get_or_load(model_name)
```

### Slow Loading

**Issue**: Reference model loading takes too long

**Solutions**:
1. Use smaller reference model
2. Pre-load reference model before training
3. Keep model in cache across runs

```python
# Pre-load before training starts
cache = get_global_reference_cache()
cache.get_or_load("meta-llama/Llama-3.2-1B")

# Then start training (cache hit, instant)
await training_client.forward_backward(...)
```

## API Reference

### ReferenceModelCache

```python
class ReferenceModelCache:
    def __init__(
        self,
        max_models: int = 2,
        quantize_by_default: bool = True,
        device: str = "cuda",
    )
    
    def get_or_load(
        self,
        model_name: str,
        quantize: Optional[bool] = None,
        load_in_4bit: bool = False,
    ) -> Any
    
    def contains(self, model_name: str) -> bool
    def remove(self, model_name: str) -> bool
    def clear(self) -> None
    def get_cache_info(self) -> Dict[str, Any]
```

### ReferenceModelService

```python
@modal.method()
def compute_log_probs(
    model_name: str,
    input_ids: List[List[int]],
    attention_mask: Optional[List[List[int]]] = None,
) -> Dict[str, Any]

@modal.method()
def compute_rewards(
    model_name: str,
    input_ids: List[List[int]],
) -> Dict[str, Any]
```

## See Also

- [RL Algorithms](RL-Algorithms.md)
- [Futures Architecture](Futures-Architecture.md)
- [Metrics & Monitoring](Metrics.md)

