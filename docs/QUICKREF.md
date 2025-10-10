# Quick Reference - Signal

## Installation

```bash
# Setup
pip install -r requirements.txt
modal setup
modal secret create secrets-hf-wandb HUGGINGFACE_TOKEN=hf_...
modal volume create signal-data

# Deploy
modal deploy modal_runtime/primitives.py

# Start API
python api/main.py
```

## API Key Management

```bash
# Generate key
python scripts/manage_keys.py generate USER_ID --description "Dev key"

# List keys
python scripts/manage_keys.py list USER_ID

# Revoke key
python scripts/manage_keys.py revoke sk-...
```

## Python Client

```python
from client.python_client import SignalClient

# Initialize
client = SignalClient(api_key="sk-...", base_url="http://localhost:8000")

# List models
models = client.list_models()

# Create run
run = client.create_run(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=32,
    lora_alpha=64,
    learning_rate=3e-4,
)

# Training loop
batch = [{"text": "Example text"}]
result = run.forward_backward(batch=batch)
run.optim_step()

# Sample
samples = run.sample(prompts=["Hello"], temperature=0.7)

# Save
artifact = run.save_state(mode="adapter")
```

## API Endpoints

```bash
# Create run
curl -X POST http://localhost:8000/runs \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"base_model": "meta-llama/Llama-3.2-3B", "lora_r": 32}'

# Forward-backward
curl -X POST http://localhost:8000/runs/{RUN_ID}/forward_backward \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"batch_data": [{"text": "Example"}]}'

# Optimizer step
curl -X POST http://localhost:8000/runs/{RUN_ID}/optim_step \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{}'

# Sample
curl -X POST http://localhost:8000/runs/{RUN_ID}/sample \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello"], "max_tokens": 50}'

# Save state
curl -X POST http://localhost:8000/runs/{RUN_ID}/save_state \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"mode": "adapter"}'

# Get status
curl http://localhost:8000/runs/{RUN_ID}/status \
  -H "Authorization: Bearer sk-..."
```

## Modal Commands

```bash
# Deploy functions
modal deploy modal_runtime/primitives.py

# View logs
modal app logs signal

# List volumes
modal volume list

# Check volume contents
modal volume get signal-data --path /data

# Download from volume
modal volume get signal-data --path /data/runs --output ./local_backup
```

## Supported Models

### Llama
- meta-llama/Llama-3.2-1B (L40S)
- meta-llama/Llama-3.2-3B (L40S)
- meta-llama/Llama-3.1-8B (A100-80GB)
- meta-llama/Llama-3.1-70B (4×A100-80GB)

### Gemma
- google/gemma-2-2b (L40S)
- google/gemma-2-9b (A100-80GB)

### Qwen
- Qwen/Qwen2.5-3B (L40S)
- Qwen/Qwen2.5-7B (A100-80GB)
- Qwen/Qwen2.5-32B (4×A100-80GB)

## Configuration

### LoRA Settings
```python
lora_r=32              # Rank (higher = more capacity)
lora_alpha=64          # Scaling (usually 2×rank)
lora_dropout=0.0       # Dropout rate
lora_target_modules=None  # Auto-select or specify list
```

### Training Settings
```python
optimizer="adamw_8bit"      # or "adamw"
learning_rate=3e-4
weight_decay=0.01
max_seq_length=2048
bf16=True
gradient_checkpointing=True
load_in_8bit=True          # 8-bit quantization
```

## File Locations

### Run Artifacts
```
/data/runs/{user_id}/{run_id}/
├── config.json
├── lora_adapters/
│   ├── step_0/
│   └── step_N/
├── optimizer_state.pt
├── gradients/
│   └── step_N.pt
└── checkpoints/
```

### API Keys
```
/data/api_keys.json
```

### Run Registry
```
/data/runs_registry.json
```

## Common Issues

### "Function not found"
```bash
modal deploy modal_runtime/primitives.py
```

### "Invalid API key"
```bash
python scripts/manage_keys.py generate NEW_USER
```

### GPU OOM
```python
# Reduce memory usage
run = client.create_run(
    max_seq_length=1024,  # Reduce from 2048
    load_in_8bit=True,
)
```

### Volume access issues
```bash
modal volume list
modal volume create signal-data  # If missing
```

## Environment Variables

```bash
# .env file
API_KEY=sk-...
BASE_URL=http://localhost:8000
HUGGINGFACE_TOKEN=hf_...
WANDB_API_KEY=...  # Optional
```

## Development

```bash
# Run API locally with auto-reload
uvicorn api.main:app --reload

# Test a single function
modal run modal_runtime/primitives.py::create_run

# Shell into Modal container
modal shell modal_runtime/primitives.py
```

## Monitoring

```bash
# Check API health
curl http://localhost:8000/health

# View app logs
modal app logs signal

# Check storage
modal volume get signal-data
```

## Cost Estimates

- L40S: ~$0.60/hour
- A100-80GB: ~$2/hour  
- 4×A100-80GB: ~$8/hour
- Storage: ~$0.10/GB/month

## Support

- Docs: ../README.md
- Deployment: DEPLOYMENT.md
- Examples: ../examples/sft_example.py
- GitHub Issues: [Your repo URL]
