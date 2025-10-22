# Signal API Complete Flow Documentation

## Overview

This document provides an excruciatingly detailed trace of what happens when a user runs the `simple_example.py` script. Every function call, API request, database operation, and Modal container interaction is documented.

## User Script Analysis

The `simple_example.py` script performs the following operations:

1. Creates a SignalClient with API key
2. Creates a training run with Llama-3.2-3B model
3. Performs 20 forward-backward passes with PPO-style training data
4. Applies optimizer steps after each forward-backward
5. Generates samples every 5 steps
6. Saves the model state
7. Repeats the process with gradient accumulation
8. Repeats again with learning rate scheduling

## Complete Flow Trace

### Phase 1: Client Initialization

#### 1.1 Script Execution Starts

```python
# examples/simple_example.py:10
client = SignalClient(api_key=API_KEY)
```

**What happens:**

- `SignalClient.__init__()` is called (`client/client.py:270-292`)
- Creates a `requests.Session()` with authentication headers
- Sets `base_url` to "https://api.frontier-signal.com"
- Sets `timeout` to 300 seconds
- Headers are set: `Authorization: Bearer {api_key}`, `Content-Type: application/json`

#### 1.2 Run Creation Request

```python
# examples/simple_example.py:11-16
run = client.create_run(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=32,
    lora_alpha=64,
    learning_rate=3e-4,
)
```

**Client-side flow:**

- `SignalClient.create_run()` is called (`client/client.py:400-453`)
- Creates `RunConfig` object with validation (`client/schemas.py:7-65`)
- Validates model name format (must contain "/")
- Validates GPU config if provided
- Calls `self._request("POST", "/runs", json=config.model_dump())`

**API Request:**

- HTTP POST to `https://api.frontier-signal.com/runs`
- Headers: `Authorization: Bearer sk-xxx`, `Content-Type: application/json`
- Body: JSON with run configuration

### Phase 2: API Server Processing

#### 2.1 Authentication (`main.py:229-275`)

- `verify_auth()` dependency is triggered
- Extracts Bearer token from Authorization header
- Calls `AuthManager.validate_api_key()` (`api/auth.py:37-120`)
- Queries Supabase `api_keys` table with key prefix for performance
- Verifies bcrypt hash of the API key
- Checks key expiration and user existence
- Updates `last_used` timestamp
- Sets user context for RLS policies
- Returns `user_id` for the authenticated user

#### 2.2 Run Creation Endpoint (`main.py:671-810`)

- `create_run()` endpoint is called
- Rate limiting: 5 runs per minute per user
- Logs run creation attempt with IP address

**Model Validation:**

- Checks if model is supported via `ModelRegistry.is_supported()`
- Loads model config from `config/models.yaml`

**GPU Allocation:**

- Calls `allocate_gpu_config()` (`api/gpu_allocator.py:131-163`)
- For Llama-3.2-3B (3.2B parameters): allocates "L40S:1"
- Validates GPU config format

**Balance Check:**

- Calculates minimum balance required (30 minutes + checkpoint buffer)
- Calls `FrontierClient.get_balance()` (`api/frontier_client.py:31-54`)
- Makes HTTP request to Frontier Backend `/internal/get-balance`
- Requires minimum balance or raises 402 Payment Required

**Run Registry:**

- Calls `RunRegistry.create_run()` (`api/registry.py:23-59`)
- Uses atomic database function `create_run_if_allowed` to prevent race conditions
- Checks concurrent run limit (max 5 per user)
- Creates run record in Supabase `runs` table
- Generates unique `run_id` (format: `run_{16-char-hex}`)

**Modal Container Initialization:**

- Calls `get_training_session()` (`main.py:70-98`)
- Looks up Modal class `signal.TrainingSession` from environment "main"
- Creates new instance (Modal handles routing by run_id)

**Modal Remote Call:**

- Calls `session.initialize.remote()` with all parameters
- This triggers Modal container startup and model loading

### Phase 3: Modal Container Startup

#### 3.1 Container Lifecycle (`modal_runtime/training_session.py:71-98`)

- `@modal.enter()` decorator triggers `container_startup()`
- Initializes `Accelerator` with bfloat16 mixed precision
- Detects available GPUs and logs their specifications
- Starts background monitoring thread for auto-checkpointing
- Sets up graceful shutdown handlers

#### 3.2 Model Loading (`modal_runtime/training_session.py:147-387`)

- `initialize()` method is called remotely
- **This is the most expensive operation (30-60 seconds)**

**Volume Setup:**

- Reloads Modal volume to get latest checkpoints
- Creates run-specific directory structure
- Stores configuration in memory

**WandB Integration (if configured):**

- Initializes WandB with experiment tracking
- Creates experiment name: `{model_name}_{timestamp}`

**Model Loading Process:**

- Calls `load_model_and_tokenizer()` (`modal_runtime/model_loader.py`)
- Downloads model from HuggingFace Hub (cached on volume)
- Loads with 4-bit quantization and bfloat16 precision
- Applies gradient checkpointing for memory efficiency
- Uses `device_map="auto"` for multi-GPU placement

**LoRA Application:**

- Calls `apply_lora_to_model()` (`modal_runtime/model_loader.py`)
- Adds LoRA adapters to specified target modules
- Configures LoRA rank (32), alpha (64), dropout (0.0)

**Optimizer Setup:**

- Creates `AdamW8bit` optimizer from bitsandbytes
- Prepares model and optimizer with Accelerate for multi-GPU
- Loads optimizer state if resuming from checkpoint

**Initial Checkpoint:**

- Saves initial checkpoint at step 0
- Commits changes to Modal volume

**Response:**

- Returns session info with parameter counts and configuration
- Model is now loaded and ready in GPU memory

### Phase 4: Training Loop Execution

#### 4.1 Forward-Backward Pass (`examples/simple_example.py:40`)

```python
fb_result = run.forward_backward(batch=training_batch)
```

**Client-side flow:**

- `SignalRun.forward_backward()` is called (`client/client.py:50-74`)
- Delegates to `SignalClient.forward_backward()` (`client/client.py:455-483`)
- Creates `TrainingClient` instance (`client/client.py:688-724`)
- Calls `TrainingClient.forward_backward()` (`client/training_client.py:108-134`)

**API Request:**

- HTTP POST to `/runs/{run_id}/forward_backward`
- Rate limiting: 1000 requests per minute
- Body contains batch data, loss function, and parameters

**API Server Processing:**

- `forward_backward()` endpoint (`main.py:813-872`)
- Verifies run ownership via `get_authorized_run()`
- Checks balance and charges incrementally (`check_and_charge_incremental()`)
- Gets GPU config from run configuration
- Calls `get_training_session()` to get Modal container

**Modal Remote Execution:**

- `session.forward_backward.remote()` (`modal_runtime/training_session.py:390-499`)
- **Model is already loaded - no loading time!**

**Training Process:**

- Tokenizes batch using model's tokenizer
- Sets model to training mode
- Moves batch to GPU device (handled by Accelerate)
- Performs forward pass through model
- Computes loss (causal language modeling)
- Performs backward pass with gradient accumulation
- Computes gradient norm for monitoring
- Updates accumulation counter

**Response:**

- Returns loss, step, gradient norm, and metrics
- Updates run registry with metrics
- Returns `ForwardBackwardResponse` to client

#### 4.2 Optimizer Step (`examples/simple_example.py:44`)

```python
run.optim_step()
```

**Client-side flow:**

- `SignalRun.optim_step()` (`client/client.py:76-91`)
- Delegates to `TrainingClient.optim_step()` (`client/training_client.py:136-151`)

**API Request:**

- HTTP POST to `/runs/{run_id}/optim_step`
- Rate limiting: 300 requests per minute

**API Server Processing:**

- `optim_step()` endpoint (`main.py:932-985`)
- Similar authentication and balance checks
- Gets Modal container session

**Modal Remote Execution:**

- `session.optim_step.remote()` (`modal_runtime/training_session.py:501-588`)
- **Optimizer is already loaded - no setup time!**

**Optimization Process:**

- Applies gradient clipping if specified
- Calls `optimizer.step()` to update parameters
- Calls `optimizer.zero_grad()` to clear gradients
- Increments step counter
- Checks for auto-checkpoint (every 100 steps)
- Logs metrics to WandB if configured
- Updates learning rate if scheduler exists

**Response:**

- Returns new step number and learning rate
- Updates run registry with step and metrics

#### 4.3 Sample Generation (`examples/simple_example.py:46-51`)

```python
_ = run.sample(
    prompts=["The meaning of life is"],
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
)
```

**Client-side flow:**

- `SignalRun.sample()` (`client/client.py:93-120`)
- Delegates to `InferenceClient.sample()` (`client/inference_client.py:123-162`)

**API Request:**

- HTTP POST to `/runs/{run_id}/sample`
- Rate limiting: 20 requests per minute

**API Server Processing:**

- `sample()` endpoint (`main.py:1039-1089`)
- Similar authentication and balance checks
- Uses single GPU for inference regardless of training GPU count

**Modal Remote Execution:**

- `session.sample.remote()` (`modal_runtime/training_session.py:590-683`)
- **Model is already loaded - no loading time!**

**Generation Process:**

- Unwraps model from Accelerate wrapper
- Sets model to evaluation mode
- For each prompt:
  - Tokenizes input prompt
  - Generates tokens using model.generate()
  - Decodes generated tokens to text
- Switches model back to training mode

**Response:**

- Returns list of generated text completions
- Includes token IDs and log probabilities if requested

### Phase 5: Model State Saving

#### 5.1 Save State (`examples/simple_example.py:53`)

```python
_ = run.save_state(mode="adapter")
```

**Client-side flow:**

- `SignalRun.save_state()` (`client/client.py:122-143`)
- Delegates to `TrainingClient.save_checkpoint()` (`client/training_client.py:231-245`)

**API Request:**

- HTTP POST to `/runs/{run_id}/save_state`
- Rate limiting: 5 requests per minute

**API Server Processing:**

- `save_state()` endpoint (`main.py:1263-1334`)
- Similar authentication and balance checks
- Uses single GPU for save operation

**Modal Remote Execution:**

- `session.save_state.remote()` (`modal_runtime/training_session.py:685-818`)
- **Model is already loaded - no loading time!**

**Save Process:**

- Unwraps model from Accelerate wrapper
- Saves LoRA adapters using PEFT's `save_pretrained()`
- Saves tokenizer configuration
- Saves optimizer state to disk
- Uploads checkpoint to S3/R2 storage
- Generates signed download URL (1-hour expiration)
- Optionally pushes to HuggingFace Hub
- Commits changes to Modal volume

**Database Updates:**

- Records artifact in `artifacts` table
- Updates run's S3 URI
- Tracks file sizes and metadata

**Response:**

- Returns save paths, S3 URI, and download URL
- Includes manifest with file information

### Phase 6: Gradient Accumulation Loop

#### 6.1 Accumulation Training (`examples/simple_example.py:63-75`)

```python
for step in range(20):
    fb_result = run.forward_backward(batch=training_batch, accumulate=True)
    # ... rest of loop
```

**Key Difference:**

- `accumulate=True` parameter is passed
- Gradients are accumulated instead of replaced
- Same Modal container is reused (no model reloading)
- All other operations remain identical

### Phase 7: Learning Rate Scheduling Loop

#### 7.1 Final Training Loop (`examples/simple_example.py:78-90`)

```python
for step in range(20):
    fb_result = run.forward_backward(batch=training_batch)
    # ... rest of loop
```

**Key Difference:**

- Learning rate can be overridden in `optim_step()`
- Same Modal container continues to be used
- All operations remain identical to first loop

## Key Performance Characteristics

### Cold Start vs Warm Operations

- **Cold Start (first call)**: 30-60 seconds for model loading
- **Warm Operations**: 1-5 seconds for forward-backward, <1 second for optimizer step
- **Sample Generation**: 2-10 seconds depending on max_tokens
- **Save State**: 10-30 seconds depending on model size and S3 upload

### Resource Utilization

- **GPU Memory**: Model stays loaded throughout entire session
- **CPU**: Minimal for warm operations
- **Network**: Only for API requests and S3 uploads
- **Storage**: Persistent volume for checkpoints and cache

### Cost Implications

- **GPU Time**: Charged per hour based on GPU type (L40S: $2/hr)
- **Storage**: Charged per GB-month for artifacts
- **Incremental Billing**: Charged every 2 minutes or $1 increments
- **Balance Checks**: Prevents overspending

## Error Handling and Recovery

### Authentication Failures

- Invalid API keys return 401 Unauthorized
- Expired keys are rejected
- Missing authorization headers are rejected

### Resource Limits

- Concurrent run limits (5 per user)
- Rate limiting on all endpoints
- Balance checks prevent overspending

### Modal Failures

- Container crashes trigger cleanup
- Failed initialization marks run as failed
- Auto-save on container shutdown

### Network Issues

- Retry logic with exponential backoff
- Connection pooling for efficiency
- Timeout handling for long operations

## Security Considerations

### API Key Security

- Keys are hashed with bcrypt
- Prefix-based lookups for performance
- Timing attack prevention
- Automatic expiration support

### Data Isolation

- Row Level Security (RLS) policies
- User context isolation
- Run ownership verification
- Secure S3 storage with signed URLs

### Resource Protection

- Rate limiting prevents abuse
- Balance checks prevent overspending
- Concurrent run limits prevent resource exhaustion
- Input validation on all parameters

## Monitoring and Observability

### Logging

- Structured logging throughout the stack
- Security event logging
- Performance metrics tracking
- Error tracking and alerting

### Metrics

- Training metrics (loss, gradient norm)
- GPU utilization monitoring
- Cost tracking and billing
- API usage statistics

### Health Checks

- Database connectivity monitoring
- Modal service availability
- External service dependencies
- Resource utilization tracking

This completes the comprehensive flow documentation for the Signal API when running the simple example script.
