# Signal - Quick Start Guide

Signal is an open-source finetuning API that makes training language models incredibly easy by handling all the infrastructure.

## What You Get

- **4 Low-Level Primitives**: Complete control over your training loop
  - `create_run` - Initialize model with LoRA adapters
  - `forward_backward` - Compute gradients for a batch
  - `optim_step` - Update model weights
  - `sample` - Generate text from current checkpoint
  - `save_state` - Export LoRA adapter or merged model

- **Modal Integration**: All training runs on serverless GPU infrastructure
- **Supabase Backend**: Runs and metrics stored in PostgreSQL
- **WandB & HuggingFace**: Optional integrations for experiment tracking and model hosting

## Prerequisites

1. **Supabase Account** - For auth and database ([supabase.com](https://supabase.com))
2. **Modal Account** - For GPU infrastructure ([modal.com](https://modal.com))
3. **Python 3.12+**

## Setup

### 1. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# Supabase (get from project settings)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Optional: for local dev
SUPABASE_ANON_KEY=your-anon-key
```

### 2. Setup Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal setup

# Create a secret for HuggingFace (required for model access)
modal secret create secrets-hf-wandb HF_TOKEN=your_huggingface_token

# Optional: Add WandB key to same secret
# modal secret create secrets-hf-wandb WANDB_API_KEY=your_wandb_key
```

### 3. Deploy Modal Functions

```bash
cd signal
modal deploy modal_runtime/primitives.py
```

You should see:
```
✓ Created function create_run
✓ Created function forward_backward
✓ Created function optim_step
✓ Created function sample
✓ Created function save_state
```

### 4. Setup Supabase Database

The Frontier project already has the necessary tables. If setting up from scratch, run the migrations:

```sql
-- Tables needed: profiles, runs, run_metrics, user_credits, api_keys, user_integrations
-- See Frontier/frontend/supabase/migrations/ for full schema
```

### 5. Start the API Server

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Usage

### Option 1: Python SDK (Recommended)

```python
from client.frontier_signal import SignalClient

# Initialize client
client = SignalClient(
    api_key="sk-your-api-key",  # Get from Supabase api_keys table
    base_url="http://localhost:8000"
)

# List available models
models = client.list_models()

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

# Save final model
artifact = run.save_state(mode="adapter", push_to_hub=False)
print(f"Saved to: {artifact['checkpoint_path']}")
```

### Option 2: Direct API Calls

```bash
# Get API key from Supabase
export API_KEY="sk-your-key"

# Create run
curl -X POST http://localhost:8000/runs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "meta-llama/Llama-3.2-3B",
    "lora_r": 32,
    "learning_rate": 3e-4
  }'

# Forward-backward
curl -X POST http://localhost:8000/runs/{run_id}/forward_backward \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_data": [{"text": "Hello world"}],
    "loss_fn": "causal_lm"
  }'
```

## Supported Models

See `config/models.yaml` for full list. Includes:
- Llama 3.2 (1B, 3B)
- Llama 3.1 (8B, 70B)
- Gemma 2 (2B, 9B)
- Qwen 2.5 (3B, 7B, 14B, 32B)

## Architecture

```
┌─────────────────┐
│  Frontend (UI)  │  - Next.js app at /signal
│  Frontier/      │  - Displays runs & metrics
└────────┬────────┘  - Reads directly from Supabase
         │
         │ (Optional: can also call API)
         │
┌────────▼────────┐
│   Signal API    │  - FastAPI server
│   signal/api/   │  - Authenticates users
└────────┬────────┘  - Calls Modal functions
         │           - Writes to Supabase
         │
┌────────▼────────┐
│ Modal Functions │  - GPU training
│ modal_runtime/  │  - Model loading
└─────────────────┘  - LoRA fine-tuning

┌─────────────────┐
│   Supabase DB   │  - Stores runs & metrics
│   PostgreSQL    │  - User authentication
└─────────────────┘  - API keys
```

## Monitoring

- **Supabase Dashboard**: View runs table and metrics in real-time
- **Frontier UI**: Visit `/signal` to see runs with charts
- **Modal Dashboard**: View function logs and GPU usage
- **WandB** (optional): Set integration in Frontier UI

## Troubleshooting

**Modal deployment fails:**
- Check `modal secret list` includes `secrets-hf-wandb`
- Verify HF token is valid for model access

**API returns 401:**
- Generate API key via `signal/scripts/manage_keys.py`
- Or create directly in Supabase `api_keys` table

**Run not appearing in frontend:**
- Check Supabase `runs` table for the run_id
- Verify user_id matches the authenticated user
- Check RLS policies allow reading own runs

**GPU costs:**
- Small models (1-3B): ~$0.60/hr on L40S
- Large models (70B): ~$8/hr on 4×A100-80GB
- Use spot instances for 50% discount

## Next Steps

- Add your HuggingFace token to user_integrations for model pushing
- Add WandB key for experiment tracking
- Customize models in `config/models.yaml`
- Build training loops with the SDK

## Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/signal/issues)
- Documentation: See `/signal/docs/` for detailed guides
- Modal Docs: [modal.com/docs](https://modal.com/docs)

