# Signal - Training API for Language Models

Fine-tune language models with LoRA using a simple, powerful API. Train models with composable primitives that give you full control over the training loop.

## Overview

Signal exposes a clean, low-level training API based on four primitives:

1. **`forward_backward`** - Compute gradients for a batch
2. **`optim_step`** - Update model weights
3. **`sample`** - Generate text from current checkpoint
4. **`save_state`** - Export LoRA adapter or merged model

This design gives you full control over the training loop while we handle the infrastructure. Inspired by pioneering work from Thinking Machines on [LoRA fine-tuning](https://thinkingmachines.ai/blog/lora).

### Train Your First Model

```python
import rewardsignal

# Initialize client
client = SignalClient(
    api_key="sk-...",  # Your API key
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
    {"input": "The quick brown fox jumps over the lazy dog."},
    {"output": "Machine learning is transforming technology."},
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

## Supported Models

### Llama Family

- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B`
- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B`
- `meta-llama/Llama-3.3-70B-Instruct`

### Gemma Family

- `google/gemma-2-2b`
- `google/gemma-2-9b`
- `unsloth/gemma-2-9b-it-bnb-4bit`

### Qwen Family

- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-14B`
- `Qwen/Qwen2.5-32B`

Want a specific model? Contact us and we'll add it!

## Automatic GPU Allocation

Signal automatically allocates the optimal GPU resources based on your model size. You don't need to think about infrastructure - just specify your model and Signal handles the rest:

### Allocation Rules

- **< 1B parameters**: L40S (single GPU)
- **1B - 7B parameters**: L40S or A100 (single GPU)
- **7B - 13B parameters**: A100-80GB (single GPU)
- **13B - 30B parameters**: A100-80GB (2 GPUs)
- **30B - 70B parameters**: A100-80GB (4 GPUs)
- **> 70B parameters**: A100-80GB (8 GPUs) or H100 (4 GPUs)

### Single-GPU Training (Small Models)

Models under 7B parameters automatically use single-GPU training:

- Efficient LoRA fine-tuning
- Fast iteration speed
- Cost-effective for smaller models
- Optional quantization (4-bit/8-bit)

### Multi-GPU Training (Large Models)

Models over 7B parameters automatically use multi-GPU training with DataParallel:

- Distributed across 2-8 GPUs
- Transparent to your training code
- Same API primitives work seamlessly
- Quantization disabled (incompatible with DataParallel)

```python
# Automatic allocation - no GPU config needed!
run = client.create_run(
    base_model="Qwen/Qwen2.5-7B",  # Automatically uses optimal GPU config
    lora_r=32,
)

# Training loop is identical regardless of GPU count
result = run.forward_backward(batch=batch)
run.optim_step()
```

### Override GPU Allocation

You can optionally override the automatic allocation for specific use cases:

```python
# Automatic allocation (recommended)
run = client.create_run(
    base_model="Qwen/Qwen2.5-7B",
    lora_r=32,
)

# Manual override - use 2x L40S instead
run = client.create_run(
    base_model="Qwen/Qwen2.5-7B",
    gpu_config="L40S:2",  # Override auto-allocation
    lora_r=32,
)

# Use H100 for maximum performance
run = client.create_run(
    base_model="meta-llama/Llama-3.1-70B",
    gpu_config="H100:4",  # Use 4x H100 for fastest training
    lora_r=32,
)
```

**Supported GPU types:**

- **L40S** - NVIDIA L40S (48GB) - Cost-effective for most models
- **A100** - NVIDIA A100 (40GB) - Good balance of performance and cost
- **A100-80GB** - NVIDIA A100 (80GB) - For larger models and multi-GPU
- **H100** - NVIDIA H100 (80GB) - Maximum performance
- **T4** - NVIDIA T4 (16GB) - Budget option for small models
- **A10G** - NVIDIA A10G (24GB) - Good for small-medium models

## Custom LoRA Configuration

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

## Setup

Signal is designed to be self-hosted on your own infrastructure.

### Prerequisites

1. **Supabase Account** - Sign up at [supabase.com](https://supabase.com) for auth and database
2. **Modal Account** - Sign up at [modal.com](https://modal.com) for GPU infrastructure
3. **HuggingFace Account** - For model access tokens
4. **Python 3.12+** - Recommended version

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/signal.git
cd signal

# Install dependencies
pip install -r requirements.txt

# Setup Supabase (follow docs/SUPABASE_SETUP.md)
# 1. Create Supabase project
# 2. Run SQL migrations from docs/SUPABASE_SETUP.md
# 3. Configure Google OAuth

# Configure environment variables
cp .env.example .env
# Edit .env with your Supabase credentials

# Setup Modal
modal setup

# Deploy Modal functions
modal deploy modal_runtime/primitives.py

# Create API key in Supabase api_keys table or via script
# (See QUICKSTART.md for details)
```

See [QUICKSTART.md](./QUICKSTART.md) for detailed setup instructions.

### Start the API Server

```bash
# Run locally
python api/main.py

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Use the API

```python
from rewardsignal import SignalClient

client = SignalClient(
    api_key="sk-...",  # Your API key
    base_url="http://localhost:8000"
)

run = client.create_run(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=32,
)
```

### Configuration

Edit `config/models.yaml` to add or modify supported models:

```yaml
models:
  - name: "your-org/your-model"
    framework: "transformers"  # or "unsloth"
    gpu: "a100-80gb:4"  # Specify GPU type and count
    family: "llama"
```

**GPU Configuration Format:**

- Single GPU: `"l40s:1"`, `"a100-80gb:1"`, `"h100:1"`
- Multi-GPU: `"a100-80gb:4"`, `"h100:8"`
- The system automatically uses FSDP for multi-GPU setups

### Technical Architecture

**Training Infrastructure:**

Signal uses a hybrid approach for maximum efficiency:

- **Single-GPU (1 GPU)**: Direct PEFT with 8-bit quantization
  - Faster for small models (1-8B parameters)
  - Lower memory overhead
  - Uses `bitsandbytes` for quantization

- **Multi-GPU (2-8 GPUs)**: Accelerate with FSDP
  - Required for large models (70B+ parameters)
  - Fully Sharded Data Parallel across all GPUs
  - Automatic model sharding and gradient synchronization
  - Mixed precision (bfloat16) training

**Modal GPU Allocation:**

- GPU resources are allocated dynamically per run
- Uses Modal's `with_options()` for runtime GPU selection
- Each training primitive (`forward_backward`, `optim_step`) runs on the same GPU configuration
- Inference always uses single GPU for efficiency

**How It Works:**

When configured, Signal API will:

1. **Before training**: Validate user has sufficient credits via `/internal/validate-credits`
2. **Before training**: Fetch user integrations (WandB, HuggingFace keys) via `/internal/get-integrations`
3. **After completion**: Deduct credits via `/internal/deduct-credits`

**Self-Hosting Without Billing:**

Leave `FRONTIER_BACKEND_URL` empty to run Signal without credit management. All training operations will work normally, but credit validation and integration management will be disabled.

### Storage

**Database (Supabase PostgreSQL):**

- User profiles and authentication
- API keys (hashed)
- Run metadata and configuration
- Training metrics and logs

**Files (Modal Volumes):**

```bash
/data/runs/{user_id}/{run_id}/
  ├── config.json           # Run configuration
  ├── lora_adapters/        # LoRA checkpoints by step
  ├── optimizer_state.pt    # Optimizer state
  ├── gradients/            # Saved gradients
  └── checkpoints/          # Exported checkpoints
```

### For Self-Hosters

When setting up your Supabase database:

1. Run the RLS migration: `supabase/migrations/20250110000001_rls_security.sql`
2. Use `SUPABASE_ANON_KEY` in your `.env` (not service role key)
3. RLS policies automatically protect your data

See [`supabase/SETUP.md`](supabase/SETUP.md) for detailed migration instructions.

## Quick Start

See [QUICKSTART.md](./QUICKSTART.md) for detailed setup instructions.

### TL;DR

```bash
# 1. Setup Modal
modal setup
modal secret create secrets-hf-wandb HF_TOKEN=your_hf_token

# 2. Deploy functions
modal deploy modal_runtime/primitives.py

# 3. Start API (in another terminal)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Use the SDK
```

## Contributing

We adore contributions!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: See `QUICKSTART.md` and `docs/` folder

## Acknowledgments

Built with:

- [Modal](https://modal.com/) - Serverless GPU infrastructure
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers) - Model library
- [PEFT](https://github.com/huggingface/peft) - LoRA implementation
- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning

---

Made with ❤️ for the AI research and engineer community
