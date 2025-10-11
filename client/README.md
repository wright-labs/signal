# Frontier Signal Python SDK

Official Python SDK for the Frontier Signal training API - fine-tune language models with LoRA using simple, powerful primitives.

## Installation

```bash
pip install frontier-signal
```

For development:
```bash
pip install frontier-signal[dev]
```

## Quick Start

### Synchronous Client

```python
from frontier_signal import SignalClient

# Initialize client
client = SignalClient(
    api_key="sk-...",  # Your API key
    base_url="https://api.frontier-signal.com"
)

# List available models
models = client.list_models()
print(f"Available models: {models}")

# Create a training run
run = client.create_run(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=32,
    lora_alpha=64,
    learning_rate=3e-4,
)

# Prepare training data
batch = [
    {"text": "The quick brown fox jumps over the lazy dog."},
    {"text": "Machine learning is transforming technology."},
]

# Training loop
for step in range(10):
    # Forward-backward pass
    result = run.forward_backward(batch=batch)
    print(f"Step {step}: Loss = {result['loss']:.4f}")
    
    # Optimizer step
    run.optim_step()
    
    # Sample from model every 5 steps
    if step % 5 == 0:
        samples = run.sample(
            prompts=["The meaning of life is"],
            temperature=0.7,
        )
        print(f"Sample: {samples['outputs'][0]}")

# Save final model
artifact = run.save_state(mode="adapter", push_to_hub=False)
print(f"Saved to: {artifact['checkpoint_path']}")
```

### Asynchronous Client

```python
import asyncio
from frontier_signal import AsyncSignalClient

async def train():
    # Use async context manager for automatic cleanup
    async with AsyncSignalClient(
        api_key="sk-...",
        base_url="https://api.frontier-signal.com"
    ) as client:
        # Create run
        run = await client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            lora_r=32,
        )
        
        # Training data
        batch = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Machine learning is transforming technology."},
        ]
        
        # Training loop
        for step in range(10):
            result = await run.forward_backward(batch=batch)
            print(f"Step {step}: Loss = {result['loss']:.4f}")
            
            await run.optim_step()
        
        # Save model
        artifact = await run.save_state(mode="adapter")
        print(f"Saved to: {artifact['checkpoint_path']}")

# Run the async function
asyncio.run(train())
```

### Context Manager (Sync)

```python
from frontier_signal import SignalClient

with SignalClient(api_key="sk-...") as client:
    run = client.create_run(base_model="meta-llama/Llama-3.2-3B")
    # Training code here...
```

## Features

- **Sync & Async Support** - Use `SignalClient` for synchronous code or `AsyncSignalClient` for async/await
- **Progressive API** - Simple API for beginners, advanced specialized clients for production
- **Type Hints** - Full type annotations for better IDE support and type checking
- **Custom Exceptions** - Specific exceptions for different error types (auth, rate limits, etc.)
- **Context Managers** - Automatic resource cleanup with context managers
- **Pydantic Models** - Request/response validation with Pydantic schemas
- **Specialized Clients** - Separate TrainingClient and InferenceClient for advanced use cases

## API Reference

### Client Initialization

```python
SignalClient(
    api_key: str,
    base_url: str = "https://api.frontier-signal.com",
    timeout: int = 300,
)
```

**Parameters:**
- `api_key` - Your Frontier Signal API key (starts with `sk-`)
- `base_url` - API server URL (default: production)
- `timeout` - Request timeout in seconds (default: 300)

### Creating a Run

```python
run = client.create_run(
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
)
```

**Returns:** `SignalRun` object with methods for training

### Training Methods

#### Forward-Backward Pass

```python
result = run.forward_backward(
    batch: List[Dict[str, Any]],
    accumulate: bool = False,
)
```

**Batch format:**
- Text: `{"text": "Your text here"}`
- Chat: `{"messages": [{"role": "user", "content": "Hello"}]}`

**Returns:**
```python
{
    "loss": 0.5,
    "step": 1,
    "grad_norm": 0.25,
    "grad_stats": {...}
}
```

#### Optimizer Step

```python
result = run.optim_step(
    learning_rate: Optional[float] = None,
)
```

**Returns:**
```python
{
    "step": 1,
    "learning_rate": 0.0003,
    "metrics": {...}
}
```

#### Sample/Generate

```python
result = run.sample(
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    return_logprobs: bool = False,
)
```

**Returns:**
```python
{
    "outputs": ["Generated text..."],
    "logprobs": [...] # if return_logprobs=True
}
```

#### Save State

```python
result = run.save_state(
    mode: Literal["adapter", "merged"] = "adapter",
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
)
```

**Returns:**
```python
{
    "artifact_uri": "s3://...",
    "checkpoint_path": "/data/...",
    "pushed_to_hub": False,
    "hub_model_id": None
}
```

### Run Information

```python
# Get current status
status = run.get_status()

# Get metrics history
metrics = run.get_metrics()
```

### List Operations

```python
# List all models
models = client.list_models()

# List all runs
runs = client.list_runs()
```

## API Levels

The SDK provides **three levels of API** to match your expertise and needs:

### Level 1: Simple API (Recommended for Beginners)

The simple API provides direct methods on the client for common operations:

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

# Direct training calls
client.forward_backward(run.run_id, batch)
client.optim_step(run.run_id)
client.sample(run.run_id, prompts)
```

### Level 2: Advanced Training API

For production training, use the specialized `TrainingClient`:

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B")

# Get specialized training client
training = client.training(
    run_id=run.run_id,
    timeout=7200,  # 2 hours for long training
    max_retries=3,
)

# Fine-grained control
for batch in dataloader:
    result = training.forward_backward(batch)
    
    # Conditional optimizer step (e.g., gradient clipping)
    if result['grad_norm'] < 10.0:
        training.optim_step()

# Convenience methods
training.train_batch(batch)  # forward_backward + optim_step
training.train_epoch(dataloader, progress=True)  # Full epoch with progress bar

# State tracking
metrics = training.get_metrics()
print(f"Average loss: {metrics['avg_loss']:.4f}")

# Save checkpoint
training.save_checkpoint(mode="adapter")
```

**Features:**
- Training-optimized defaults (1 hour timeout, exponential backoff)
- State tracking (loss history, gradient norms)
- Convenience methods (train_batch, train_epoch)
- Context manager support

### Level 3: Advanced Inference API

For production inference, use the specialized `InferenceClient`:

```python
from frontier_signal import SignalClient

client = SignalClient(api_key="sk-...")

# Get specialized inference client
inference = client.inference(
    run_id="run_123",
    step=100,  # Use specific checkpoint
    batch_size=32,  # Batch size for inference
    timeout=30,
)

# Batched generation (automatic chunking)
prompts = ["Hello"] * 100
outputs = inference.batch_sample(
    prompts=prompts,
    max_tokens=50,
)

# Enable caching for repeated prompts
inference.enable_cache()
outputs = inference.sample(["Same prompt"], max_tokens=50)  # Cached on repeat

# Compare different checkpoints
inference_early = client.inference(run_id, step=10)
inference_late = client.inference(run_id, step=1000)
```

**Features:**
- Inference-optimized defaults (30s timeout, immediate retry)
- Automatic batching for efficiency
- Response caching
- Future: streaming, embeddings

**📚 For detailed examples and comparisons, see [HYBRID_CLIENT_GUIDE.md](HYBRID_CLIENT_GUIDE.md)**

## Advanced Usage

### Chat Templates

For instruction-tuned models:

```python
batch = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
]

run.forward_backward(batch=batch)
```

### Gradient Accumulation

Accumulate gradients across multiple batches:

```python
# Accumulate gradients
run.forward_backward(batch=batch1, accumulate=False)  # Reset
run.forward_backward(batch=batch2, accumulate=True)   # Accumulate
run.forward_backward(batch=batch3, accumulate=True)   # Accumulate

# Apply accumulated gradients
run.optim_step()
```

### Custom LoRA Configuration

```python
run = client.create_run(
    base_model="meta-llama/Llama-3.1-8B",
    lora_r=64,  # Higher rank = more capacity
    lora_alpha=128,  # Usually 2x lora_r
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
```

### Export to HuggingFace Hub

```python
artifact = run.save_state(
    mode="merged",
    push_to_hub=True,
    hub_model_id="your-username/your-model-name",
)
```

## Exception Handling

The SDK provides specific exceptions for different error types:

```python
from frontier_signal import (
    SignalAPIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)

try:
    run = client.create_run(base_model="invalid/model")
except AuthenticationError:
    print("Invalid API key")
except ValidationError as e:
    print(f"Invalid parameters: {e.message}")
except RateLimitError:
    print("Rate limit exceeded, try again later")
except NotFoundError:
    print("Resource not found")
except SignalAPIError as e:
    print(f"API error: {e.message} (status: {e.status_code})")
```

## Type Hints

The SDK includes full type annotations. Import schemas for type hints:

```python
from frontier_signal import RunConfig, TrainingExample

def prepare_batch(texts: List[str]) -> List[TrainingExample]:
    return [TrainingExample(text=t) for t in texts]

config = RunConfig(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=32,
)
```

## Development

### Local Installation

Clone the monorepo and install in editable mode:

```bash
git clone https://github.com/yourusername/frontier-signal.git
cd frontier-signal/client
pip install -e .
```

### Running Tests

```bash
cd client
pytest
```

### Building the Package

```bash
cd client
python -m build
```

### Publishing to PyPI

```bash
cd client

# Test on TestPyPI first
python -m twine upload --repository testpypi dist/*

# Then publish to PyPI
python -m twine upload dist/*
```

## Examples

See the [examples/](examples/) directory for more usage examples:
- `basic_sync.py` - Synchronous client example
- `basic_async.py` - Asynchronous client example
- `advanced_training.py` - Advanced training with TrainingClient
- `advanced_inference.py` - Advanced inference with InferenceClient

## Guides

- **[Hybrid Client Guide](HYBRID_CLIENT_GUIDE.md)** - Complete guide to all three API levels with examples

## Support

- **Documentation**: https://docs.frontier-signal.com
- **GitHub Issues**: https://github.com/yourusername/frontier-signal/issues
- **Email**: support@frontier-signal.com

## License

MIT License - see [LICENSE](../LICENSE) file for details.
