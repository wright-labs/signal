# Hybrid Client Architecture Guide

The Frontier Signal SDK provides three levels of API for different use cases: **Simple API**, **Advanced Training API**, and **Advanced Inference API**. This guide explains when and how to use each.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Simple API (Recommended for Beginners)](#simple-api)
3. [Advanced Training API](#advanced-training-api)
4. [Advanced Inference API](#advanced-inference-api)
5. [Async Support](#async-support)
6. [Migration Examples](#migration-examples)
7. [Best Practices](#best-practices)

---

## Architecture Overview

```
SignalClient (Unified API)
  ├── create_run() → SignalRun
  ├── forward_backward() → delegates to TrainingClient
  ├── optim_step() → delegates to TrainingClient
  ├── sample() → delegates to InferenceClient
  └── save_state() → delegates to TrainingClient

TrainingClient (Specialized)
  ├── forward_backward() - compute gradients
  ├── optim_step() - apply optimizer update
  ├── train_batch() - convenience wrapper (fb + optim)
  ├── train_epoch() - high-level loop with progress tracking
  ├── save_checkpoint() - save model state
  └── get_metrics() - training metrics

InferenceClient (Specialized)
  ├── sample() - generate text
  ├── batch_sample() - batched inference
  ├── stream_sample() - streaming (future)
  └── embeddings() - get embeddings (future)
```

**Key Insight**: All APIs use the same backend. The specialized clients provide:
- **Optimized defaults** (timeouts, retries)
- **Convenience methods** (train_batch, train_epoch, batch_sample)
- **State tracking** (loss history, gradient norms)
- **Better type safety** (clear separation of concerns)

---

## Simple API

**When to use**: Getting started, simple training scripts, prototyping

**Features**:
- Minimal setup
- All operations from one client
- Default timeouts and retries

### Example 1: Basic Training

```python
from frontier_signal import SignalClient

# Initialize client
client = SignalClient(api_key="sk-...")

# Create a run
run = client.create_run(
    base_model="Qwen/Qwen2.5-3B",
    lora_r=8,
    lora_alpha=16,
    learning_rate=5e-4,
)

# Train directly on client
for batch in dataloader:
    # Forward-backward pass
    result = client.forward_backward(
        run_id=run.run_id,
        batch=batch,
    )
    print(f"Loss: {result['loss']:.4f}")
    
    # Optimizer step
    client.optim_step(run_id=run.run_id)

# Save model
client.save_state(run_id=run.run_id, mode="adapter")
```

### Example 2: Using SignalRun Wrapper

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

# Use run methods (cleaner than passing run_id)
for batch in dataloader:
    result = run.forward_backward(batch=batch)
    run.optim_step()

run.save_state(mode="adapter")
```

---

## Advanced Training API

**When to use**: Production training, fine-grained control, monitoring, custom training loops

**Features**:
- Training-optimized defaults (1 hour timeout, exponential backoff)
- State tracking (loss history, gradient norms)
- Convenience methods (train_batch, train_epoch)
- Context manager support for cleanup

### Example 1: Fine-Grained Control

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

# Get specialized training client
training = client.training(
    run_id=run.run_id,
    timeout=7200,  # 2 hours
    max_retries=3,
)

# Fine-grained control over training
for batch in dataloader:
    # Compute gradients
    result = training.forward_backward(batch)
    
    # Conditional optimizer step (e.g., gradient clipping)
    if result['grad_norm'] < 100.0:
        training.optim_step()
    else:
        print(f"Skipping step: gradient norm too high ({result['grad_norm']})")

# Access training metrics
metrics = training.get_metrics()
print(f"Average loss: {metrics['avg_loss']:.4f}")
print(f"Current step: {metrics['current_step']}")

# Save checkpoint
training.save_checkpoint(mode="adapter")
```

### Example 2: Convenience Methods

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

# Get training client from run
training = run.training(timeout=7200)

# Use train_batch (combines forward_backward + optim_step)
for batch in dataloader:
    result = training.train_batch(
        batch_data=batch,
        learning_rate=1e-4,  # Override learning rate
    )
    print(f"Step {result['step']}: loss={result['loss']:.4f}")

# Or train for full epoch with progress bar
result = training.train_epoch(
    dataloader=dataloader,
    progress=True,  # Shows tqdm progress bar
)
print(f"Epoch complete: avg_loss={result['avg_loss']:.4f}")
```

### Example 3: Context Manager

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

# Use context manager for automatic cleanup
with client.training(run.run_id) as training:
    for batch in dataloader:
        training.train_batch(batch)
    
    # Save before exiting
    training.save_checkpoint(mode="adapter")

# Session automatically closed
```

---

## Advanced Inference API

**When to use**: Production inference, batched generation, low latency, caching

**Features**:
- Inference-optimized defaults (30s timeout, immediate retry)
- Batched inference support
- Response caching
- Streaming support (future)

### Example 1: Batched Inference

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")

# Get specialized inference client
inference = client.inference(
    run_id="run_123",
    step=100,  # Use checkpoint at step 100
    batch_size=32,  # Process 32 prompts at a time
    timeout=30,
)

# Batched generation (automatically chunks into batches)
prompts = ["Hello", "World", ...] * 100  # 200 prompts
outputs = inference.batch_sample(
    prompts=prompts,
    max_tokens=50,
    temperature=0.7,
)

print(f"Generated {len(outputs)} responses")
```

### Example 2: Caching for Repeated Prompts

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
inference = client.inference(run_id="run_123", step=100)

# Enable caching
inference.enable_cache()

# First call hits the API
outputs = inference.sample(
    prompts=["What is the capital of France?"],
    max_tokens=50,
)

# Second call returns cached result (instant)
outputs = inference.sample(
    prompts=["What is the capital of France?"],
    max_tokens=50,
)

# Check cache stats
stats = inference.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")

# Clear cache when done
inference.clear_cache()
```

### Example 3: Different Checkpoints

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")

# Compare outputs at different training steps
inference_early = client.inference(run_id="run_123", step=10)
inference_late = client.inference(run_id="run_123", step=1000)

prompt = "Explain quantum computing"

output_early = inference_early.sample([prompt], max_tokens=100)
output_late = inference_late.sample([prompt], max_tokens=100)

print("Early checkpoint:", output_early[0])
print("Late checkpoint:", output_late[0])
```

---

## Async Support

All specialized clients have async versions for concurrent operations.

### Example 1: Async Training

```python
import asyncio
from frontier_signal import SignalClient, AsyncTrainingClient

async def train_async():
    client = SignalClient(api_key="sk-...")
    run = client.create_run(base_model="Qwen/Qwen2.5-3B")
    
    # Get async training client
    async with AsyncTrainingClient(
        run_id=run.run_id,
        api_key=client.api_key,
        base_url=client.base_url,
    ) as training:
        for batch in dataloader:
            result = await training.train_batch(batch)
            print(f"Loss: {result['loss']:.4f}")
        
        await training.save_checkpoint(mode="adapter")

asyncio.run(train_async())
```

### Example 2: Async Batched Inference

```python
import asyncio
from frontier_signal import AsyncInferenceClient

async def infer_batch(inference, prompts):
    return await inference.batch_sample(prompts, max_tokens=50)

async def main():
    # Create async inference client
    async with AsyncInferenceClient(
        run_id="run_123",
        api_key="sk-...",
        batch_size=32,
    ) as inference:
        # Process multiple batches concurrently
        batch1 = ["prompt1", "prompt2", ...]
        batch2 = ["prompt3", "prompt4", ...]
        batch3 = ["prompt5", "prompt6", ...]
        
        results = await asyncio.gather(
            infer_batch(inference, batch1),
            infer_batch(inference, batch2),
            infer_batch(inference, batch3),
        )
        
        print(f"Processed {sum(len(r) for r in results)} prompts concurrently")

asyncio.run(main())
```

---

## Migration Examples

### From Simple to Advanced Training

**Before (Simple API)**:
```python
client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

for batch in dataloader:
    client.forward_backward(run.run_id, batch)
    client.optim_step(run.run_id)
```

**After (Advanced Training API)**:
```python
client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

training = client.training(run.run_id, timeout=7200)
for batch in dataloader:
    training.train_batch(batch)  # Cleaner + automatic metrics tracking

print(f"Final avg loss: {training.get_metrics()['avg_loss']}")
```

### From Simple to Advanced Inference

**Before (Simple API)**:
```python
client = SignalClient(api_key="sk-...")
outputs = client.sample(
    run_id="run_123",
    prompts=["Hello"] * 100,
    max_tokens=50,
)
```

**After (Advanced Inference API)**:
```python
client = SignalClient(api_key="sk-...")
inference = client.inference(run_id="run_123", batch_size=32)

# Automatically batches for efficiency
outputs = inference.batch_sample(
    prompts=["Hello"] * 100,
    max_tokens=50,
)
```

---

## Best Practices

### 1. Choose the Right API Level

- **Simple API**: Prototyping, learning, simple scripts
- **Advanced Training API**: Production training, monitoring, custom loops
- **Advanced Inference API**: Production inference, batching, low latency

### 2. Use Context Managers

Always use context managers for proper resource cleanup:

```python
with client.training(run_id) as training:
    # Your training code
    pass
# Session closed automatically
```

### 3. Share Sessions

Reuse the same session for multiple clients (connection pooling):

```python
client = SignalClient(api_key="sk-...")

# These share the same connection pool
training = client.training(run_id)
inference = client.inference(run_id)
```

### 4. Monitor Training Metrics

Use the advanced training API to track progress:

```python
training = client.training(run_id)

for epoch in range(num_epochs):
    training.train_epoch(dataloader, progress=True)
    metrics = training.get_metrics()
    
    # Log to wandb, tensorboard, etc.
    wandb.log({
        "avg_loss": metrics["avg_loss"],
        "step": metrics["current_step"],
    })
```

### 5. Optimize Inference with Batching

Always batch inference requests when possible:

```python
# Instead of:
for prompt in prompts:
    output = inference.sample([prompt])

# Do this:
outputs = inference.batch_sample(prompts, batch_size=32)
```

### 6. Use Async for Concurrency

Use async clients for concurrent operations:

```python
# Train and evaluate concurrently
async def train_and_eval():
    training_task = asyncio.create_task(train_async())
    eval_task = asyncio.create_task(eval_async())
    await asyncio.gather(training_task, eval_task)
```

---

## Summary

| Feature | Simple API | Advanced Training | Advanced Inference |
|---------|-----------|-------------------|-------------------|
| **Best for** | Beginners | Production training | Production inference |
| **Setup** | Minimal | Moderate | Moderate |
| **Timeout** | Default | Configurable (default: 1h) | Configurable (default: 30s) |
| **Retries** | Default | Exponential backoff | Immediate |
| **State tracking** | No | Yes (loss, grad_norm) | Optional (caching) |
| **Batching** | Manual | Manual | Automatic |
| **Progress bars** | Manual | Built-in | N/A |
| **Type safety** | Good | Excellent | Excellent |

**Recommendation**: Start with the **Simple API** for learning and prototyping. Graduate to **Advanced APIs** when you need more control, better performance, or production deployment.

