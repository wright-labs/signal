# Deployment Guide - Signal

This guide walks through deploying the Signal API step-by-step.

## Prerequisites

1. **Modal Account**
   - Sign up at [modal.com](https://modal.com)
   - Install Modal CLI: `pip install modal`
   - Authenticate: `modal setup`

2. **HuggingFace Account**
   - Create account at [huggingface.co](https://huggingface.co)
   - Generate access token with write permissions

3. **Python Environment**
   - Python 3.12+ recommended
   - Virtual environment recommended

## Step 1: Setup Environment

```bash
# Clone/navigate to project
cd signal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Modal Secrets

Store your HuggingFace token in Modal:

```bash
modal secret create secrets-hf-wandb \
  HUGGINGFACE_TOKEN=hf_... \
  WANDB_API_KEY=...  # Optional: for WandB logging
```

## Step 3: Create Modal Volume

The volume will store all training artifacts:

```bash
modal volume create signal-data
```

## Step 4: Deploy Modal Functions

Deploy the training primitives to Modal:

```bash
# Deploy all functions
modal deploy modal_runtime/primitives.py
```

This will:
- Build the Docker images (may take 10-15 minutes first time)
- Deploy all primitive functions
- Create the Modal app

You should see output like:
```
✓ Created objects.
├── 🔨 Created app => signal
├── 🔨 Created function => create_run
├── 🔨 Created function => forward_backward
├── 🔨 Created function => optim_step
├── 🔨 Created function => sample
└── 🔨 Created function => save_state
```

## Step 5: Initialize Storage

Create the data directory structure:

```bash
# Create data directory (locally or in Modal Volume)
mkdir -p data
```

## Step 6: Generate API Keys

Generate your first API key:

```bash
python scripts/manage_keys.py generate user_001 \
  --description "Development key"
```

Save the generated API key securely!

To list keys:
```bash
python scripts/manage_keys.py list user_001
```

## Step 7: Start API Server

### Option A: Run Locally

```bash
# Run with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python api/main.py
```

The API will be available at `http://localhost:8000`

### Option B: Deploy API Server to Modal

Create a new file `api/modal_app.py`:

```python
from modal import App, Image, web_endpoint
from .main import app as fastapi_app

modal_app = App("signal-api")

image = Image.debian_slim(python_version="3.12").pip_install(
    "fastapi",
    "uvicorn",
    "pydantic",
    "pyyaml",
)

@modal_app.function(image=image)
@web_endpoint(method="GET")
def health():
    return {"status": "healthy"}

# Deploy FastAPI app
@modal_app.asgi_app()
def fastapi_endpoint():
    return fastapi_app
```

Then deploy:
```bash
modal deploy api/modal_app.py
```

## Step 8: Test the API

### Using Python Client

```python
from client.python_client import SignalClient

client = SignalClient(
    api_key="sk-...",  # Your API key
    base_url="http://localhost:8000"
)

# List available models
models = client.list_models()
print(models)

# Create a run
run = client.create_run(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=32,
)

print(f"Created run: {run.run_id}")
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Create run
curl -X POST http://localhost:8000/runs \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "meta-llama/Llama-3.2-3B",
    "lora_r": 32,
    "lora_alpha": 64,
    "learning_rate": 3e-4
  }'
```

## Step 9: Run Example Training

```bash
# Edit examples/sft_example.py with your API key
# Then run:
python examples/sft_example.py
```

## Troubleshooting

### Modal Function Not Found

If you get "Function not found" errors:
```bash
# Redeploy functions
modal deploy modal_runtime/primitives.py
```

### HuggingFace Token Issues

Verify your secret is set:
```bash
modal secret list
```

Update if needed:
```bash
modal secret create secrets-hf-wandb \
  HUGGINGFACE_TOKEN=hf_new_token
```

### GPU Out of Memory

Reduce batch size or sequence length in your training config:
```python
run = client.create_run(
    base_model="meta-llama/Llama-3.2-3B",
    max_seq_length=1024,  # Reduce from 2048
    load_in_8bit=True,     # Use 8-bit quantization
)
```

### Volume Access Issues

Ensure volume is created and mounted:
```bash
modal volume list
modal volume get signal-data
```

## Production Deployment

### 1. Use Environment Variables

Create `.env` file:
```bash
API_BASE_URL=https://your-domain.com
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

### 2. Setup Rate Limiting

Add to `api/main.py`:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/runs")
@limiter.limit("10/minute")
async def create_run(...):
    ...
```

### 3. Add Monitoring

Install Prometheus client:
```bash
pip install prometheus-client
```

Add metrics:
```python
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
request_latency = Histogram('api_request_duration_seconds', 'Request latency')
```

### 4. Setup HTTPS

Use a reverse proxy (nginx, Caddy) or deploy behind CloudFlare.

### 5. Database for Run Registry

Replace JSON storage with PostgreSQL:
```bash
pip install sqlalchemy psycopg2-binary
```

Update `api/registry.py` to use SQLAlchemy.

## Cost Optimization

### 1. Use Spot Instances

Modal supports spot instances for cost savings. Update primitives:
```python
@app.function(
    gpu="a100-80gb:1",
    spot=True,  # Use spot instances
)
```

### 2. Auto-scaling

Configure scaledown windows:
```python
@app.function(
    gpu="a100-80gb:1",
    timeout=2 * HOURS,
    container_idle_timeout=300,  # 5 minutes
)
```

### 3. Optimize Model Loading

Cache models in volume:
```python
# Pre-download models
modal run modal_runtime/download_models.py --model meta-llama/Llama-3.2-3B
```

## Monitoring

### View Modal Logs

```bash
# View app logs
modal app logs signal

# View specific function logs
modal app logs signal --function create_run
```

### Check Storage Usage

```bash
# Check volume size
modal volume get signal-data
```

### Monitor API Health

Setup health check endpoint monitoring with:
- UptimeRobot
- Pingdom
- DataDog

## Backup and Recovery

### Backup Run Data

```bash
# Download volume contents
modal volume get signal-data --path /data --output ./backup
```

### Restore from Backup

```bash
# Upload to volume
modal volume put signal-data ./backup /data
```

## Support

For issues:
1. Check Modal logs: `modal app logs signal`
2. Check API server logs
3. Verify secrets are set correctly
4. Ensure sufficient Modal credits

## Next Steps

- [ ] Setup monitoring and alerts
- [ ] Configure rate limiting
- [ ] Setup CI/CD pipeline
- [ ] Add custom models to config
- [ ] Implement custom evaluation metrics
- [ ] Setup automated backups

