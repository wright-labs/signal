# Signal Codebase Visualization

This document provides comprehensive visualizations of the Signal codebase architecture, dependencies, and structure.

## 1. High-Level Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        SDK[Signal SDK<br/>rewardsignal package]
        CLI[CLI Tools]
        WEB[Web Interface]
    end
    
    subgraph "API Layer"
        MAIN[main.py<br/>FastAPI Application]
        AUTH[api/auth.py<br/>Authentication]
        REGISTRY[api/registry.py<br/>Run Management]
        MODELS[api/models.py<br/>Model Registry]
        PRICING[api/pricing.py<br/>Cost Calculation]
        GPU_ALLOC[api/gpu_allocator.py<br/>GPU Allocation]
    end
    
    subgraph "Infrastructure Layer"
        MODAL[Modal Runtime<br/>GPU Infrastructure]
        SUPABASE[Supabase<br/>Database & Auth]
        S3[AWS S3<br/>Storage]
        HF[HuggingFace Hub<br/>Model Storage]
    end
    
    subgraph "Training Runtime"
        SESSION[TrainingSession<br/>Stateful Training]
        LOADER[Model Loader<br/>Model Loading]
        MONITOR[GPU Monitor<br/>Resource Monitoring]
        UTILS[Training Utils<br/>Tokenization, Checkpoints]
    end
    
    SDK --> MAIN
    CLI --> MAIN
    WEB --> MAIN
    
    MAIN --> AUTH
    MAIN --> REGISTRY
    MAIN --> MODELS
    MAIN --> PRICING
    MAIN --> GPU_ALLOC
    
    AUTH --> SUPABASE
    REGISTRY --> SUPABASE
    PRICING --> SUPABASE
    
    MAIN --> MODAL
    MODAL --> SESSION
    SESSION --> LOADER
    SESSION --> MONITOR
    SESSION --> UTILS
    
    SESSION --> HF
    SESSION --> S3
```

## 2. Core API Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth
    participant Registry
    participant Modal
    participant Training
    
    Client->>API: POST /runs (create_run)
    API->>Auth: verify_auth(api_key)
    Auth->>API: user_id
    API->>Registry: validate_model()
    API->>API: allocate_gpu_config()
    API->>Modal: create_training_session()
    Modal->>Training: initialize_session()
    Training->>API: session_id
    API->>Client: RunResponse
    
    Note over Client,Training: Training Loop
    Client->>API: POST /forward_backward
    API->>Modal: forward_backward(batch)
    Modal->>Training: compute_gradients()
    Training->>API: loss, gradients
    API->>Client: ForwardBackwardResponse
    
    Client->>API: POST /optim_step
    API->>Modal: optim_step()
    Modal->>Training: update_weights()
    Training->>API: success
    API->>Client: OptimStepResponse
    
    Client->>API: POST /sample
    API->>Modal: sample(prompts)
    Modal->>Training: generate_text()
    Training->>API: outputs
    API->>Client: SampleResponse
    
    Client->>API: POST /save_state
    API->>Modal: save_state(mode)
    Modal->>Training: export_checkpoint()
    Training->>API: checkpoint_path
    API->>Client: SaveStateResponse
```

## 3. Module Dependencies

```mermaid
graph TD
    subgraph "API Package"
        MAIN[main.py]
        AUTH[auth.py]
        REGISTRY[registry.py]
        MODELS[models.py]
        SCHEMAS[schemas.py]
        PRICING[pricing.py]
        GPU_ALLOC[gpu_allocator.py]
        FRONTIER[frontier_client.py]
        FUTURE_STORE[future_store.py]
        OPENAI_COMPAT[openai_compat.py]
        SUPABASE_CLIENT[supabase_client.py]
        LOGGING[logging_config.py]
    end
    
    subgraph "Modal Runtime"
        APP[app.py]
        SESSION[training_session.py]
        LOADER[model_loader.py]
        MONITOR[gpu_monitor.py]
        METRICS[metrics.py]
        CLEANUP[cleanup.py]
        S3_CLIENT[s3_client.py]
        TRL_TRAINERS[trl_trainers.py]
        
        subgraph "Utils"
            PATHS[utils/paths.py]
            CHECKPOINT[utils/checkpoint.py]
            TOKENIZATION[utils/tokenization.py]
            PREFERENCE[utils/preference_utils.py]
        end
    end
    
    subgraph "Client Package"
        CLIENT[client.py]
        ASYNC_CLIENT[async_client.py]
        INFERENCE[inference_client.py]
        TRAINING_CLIENT[training_client.py]
        FUTURES[futures.py]
        SCHEMAS_CLIENT[schemas.py]
        EXCEPTIONS[exceptions.py]
    end
    
    MAIN --> AUTH
    MAIN --> REGISTRY
    MAIN --> MODELS
    MAIN --> SCHEMAS
    MAIN --> PRICING
    MAIN --> GPU_ALLOC
    MAIN --> FRONTIER
    MAIN --> FUTURE_STORE
    MAIN --> OPENAI_COMPAT
    MAIN --> SUPABASE_CLIENT
    MAIN --> LOGGING
    
    REGISTRY --> SUPABASE_CLIENT
    AUTH --> SUPABASE_CLIENT
    PRICING --> SUPABASE_CLIENT
    
    SESSION --> APP
    SESSION --> LOADER
    SESSION --> MONITOR
    SESSION --> METRICS
    SESSION --> S3_CLIENT
    SESSION --> TRL_TRAINERS
    
    LOADER --> PATHS
    SESSION --> PATHS
    SESSION --> CHECKPOINT
    SESSION --> TOKENIZATION
    SESSION --> PREFERENCE
    
    CLIENT --> SCHEMAS_CLIENT
    CLIENT --> EXCEPTIONS
    ASYNC_CLIENT --> SCHEMAS_CLIENT
    ASYNC_CLIENT --> EXCEPTIONS
    INFERENCE_CLIENT --> CLIENT
    TRAINING_CLIENT --> CLIENT
    FUTURES --> CLIENT
```

## 4. Data Flow Architecture

```mermaid
graph LR
    subgraph "External Services"
        HF[HuggingFace Hub<br/>Model Storage]
        S3[AWS S3<br/>Checkpoint Storage]
        SUPABASE[Supabase<br/>Metadata & Auth]
        FRONTIER[Frontier Backend<br/>Billing & Credits]
    end
    
    subgraph "Signal API"
        API[FastAPI Server<br/>main.py]
        AUTH[Auth Manager<br/>JWT + API Keys]
        REGISTRY[Run Registry<br/>Session Management]
    end
    
    subgraph "Modal Infrastructure"
        MODAL[Modal Runtime<br/>GPU Containers]
        TRAINING[Training Session<br/>Stateful Training]
        VOLUMES[Modal Volumes<br/>Persistent Storage]
    end
    
    subgraph "Client SDK"
        SDK[Signal Client<br/>Python SDK]
        FUTURES[Futures API<br/>Async Operations]
    end
    
    SDK --> API
    API --> AUTH
    AUTH --> SUPABASE
    API --> REGISTRY
    REGISTRY --> SUPABASE
    
    API --> MODAL
    MODAL --> TRAINING
    TRAINING --> VOLUMES
    TRAINING --> HF
    TRAINING --> S3
    
    API --> FRONTIER
    
    SDK --> FUTURES
    FUTURES --> API
```

## 5. Training Session State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing: create_run()
    
    Initializing --> Loading: load_model()
    Loading --> Ready: model_loaded()
    
    Ready --> ForwardBackward: forward_backward()
    ForwardBackward --> Ready: gradients_computed()
    
    Ready --> OptimStep: optim_step()
    OptimStep --> Ready: weights_updated()
    
    Ready --> Sampling: sample()
    Sampling --> Ready: text_generated()
    
    Ready --> Saving: save_state()
    Saving --> Ready: checkpoint_saved()
    
    Ready --> Monitoring: auto_checkpoint()
    Monitoring --> Ready: checkpoint_completed()
    
    Ready --> [*]: session_timeout()
    Ready --> [*]: explicit_shutdown()
    
    note right of ForwardBackward
        Computes gradients
        Updates loss
        Accumulates gradients
    end note
    
    note right of OptimStep
        Updates model weights
        Advances optimizer
        Increments step counter
    end note
    
    note right of Sampling
        Generates text samples
        Uses current checkpoint
        Non-blocking operation
    end note
```

## 6. GPU Allocation Strategy

```mermaid
graph TD
    START[User Request<br/>Model + Optional GPU Config] --> CHECK{User Override<br/>Provided?}
    
    CHECK -->|Yes| VALIDATE[Validate GPU Config<br/>Format & Availability]
    CHECK -->|No| AUTO_ALLOC[Auto-Allocate Based<br/>on Model Size]
    
    VALIDATE --> PARSE[Parse GPU Config<br/>Type:Count Format]
    AUTO_ALLOC --> MODEL_SIZE[Determine Model Size<br/>Parameter Count]
    
    MODEL_SIZE --> SIZE_CHECK{Model Size}
    
    SIZE_CHECK -->|"< 1B"| L40S_SINGLE[L40S:1<br/>Single GPU]
    SIZE_CHECK -->|"1B - 7B"| L40S_A100[L40S or A100:1<br/>Single GPU]
    SIZE_CHECK -->|"7B - 13B"| A100_80GB[A100-80GB:1<br/>Single GPU]
    SIZE_CHECK -->|"13B - 30B"| A100_80GB_2[A100-80GB:2<br/>Multi-GPU]
    SIZE_CHECK -->|"30B - 70B"| A100_80GB_4[A100-80GB:4<br/>Multi-GPU]
    SIZE_CHECK -->|"> 70B"| A100_80GB_8[A100-80GB:8<br/>or H100:4]
    
    PARSE --> FINAL_CONFIG[Final GPU Config]
    L40S_SINGLE --> FINAL_CONFIG
    L40S_A100 --> FINAL_CONFIG
    A100_80GB --> FINAL_CONFIG
    A100_80GB_2 --> FINAL_CONFIG
    A100_80GB_4 --> FINAL_CONFIG
    A100_80GB_8 --> FINAL_CONFIG
    
    FINAL_CONFIG --> COST_CALC[Calculate Cost<br/>Per Hour]
    COST_CALC --> VALIDATE_BALANCE[Validate User Balance<br/>Sufficient Credits]
    VALIDATE_BALANCE --> CREATE_SESSION[Create Training Session<br/>with GPU Config]
```

## 7. File Structure Overview

```
signal/
├── api/                          # FastAPI application
│   ├── auth.py                  # Authentication & JWT
│   ├── registry.py              # Run management
│   ├── models.py                # Model registry
│   ├── schemas.py               # Pydantic models
│   ├── pricing.py               # Cost calculation
│   ├── gpu_allocator.py         # GPU allocation logic
│   ├── frontier_client.py       # Billing integration
│   ├── future_store.py          # Async operation storage
│   ├── openai_compat.py         # OpenAI-compatible API
│   ├── supabase_client.py       # Database client
│   └── logging_config.py        # Security logging
├── modal_runtime/               # Modal GPU infrastructure
│   ├── app.py                   # Modal app configuration
│   ├── training_session.py      # Stateful training session
│   ├── model_loader.py          # Model loading utilities
│   ├── gpu_monitor.py           # GPU monitoring
│   ├── metrics.py               # Training metrics
│   ├── cleanup.py               # Resource cleanup
│   ├── s3_client.py             # S3 storage client
│   ├── trl_trainers.py          # TRL training utilities
│   └── utils/                   # Training utilities
│       ├── paths.py             # Path management
│       ├── checkpoint.py        # Checkpoint handling
│       ├── tokenization.py     # Text tokenization
│       └── preference_utils.py # Preference learning
├── client/                      # Python SDK
│   ├── rewardsignal/           # SDK package
│   │   ├── client.py           # Main client
│   │   ├── async_client.py     # Async client
│   │   ├── inference_client.py # Inference operations
│   │   ├── training_client.py  # Training operations
│   │   ├── futures.py          # Async futures
│   │   ├── schemas.py          # SDK schemas
│   │   └── exceptions.py       # Custom exceptions
│   └── examples/               # Usage examples
├── config/                      # Configuration files
│   ├── models.yaml             # Supported models
│   └── pricing.yaml            # GPU pricing
├── scripts/                     # Utility scripts
│   ├── migrations/              # Database migrations
│   └── *.py                    # Setup & management scripts
├── tests/                       # Test suite
└── main.py                     # Application entry point
```

## 8. Key Design Patterns

### Stateful Training Session
- **Pattern**: Stateful Modal class with persistent state
- **Benefits**: Maintains model, optimizer, and training state across API calls
- **Implementation**: `TrainingSession` class with `@modal.enter()` lifecycle

### Composable Training Primitives
- **Pattern**: Four core primitives (forward_backward, optim_step, sample, save_state)
- **Benefits**: Full control over training loop while handling infrastructure
- **Implementation**: Each primitive is a Modal method with GPU allocation

### Automatic GPU Allocation
- **Pattern**: Model-size-based GPU allocation with user override capability
- **Benefits**: Optimal resource usage without manual configuration
- **Implementation**: `gpu_allocator.py` with size-based rules

### Async Futures API
- **Pattern**: Non-blocking operations with future-based results
- **Benefits**: Better resource utilization and user experience
- **Implementation**: `future_store.py` with Redis-like storage

### Multi-GPU Transparency
- **Pattern**: Single-GPU API that scales to multi-GPU automatically
- **Benefits**: Same API works regardless of GPU count
- **Implementation**: Accelerate library with FSDP for multi-GPU

## 9. External Integrations

```mermaid
graph LR
    subgraph "Signal Platform"
        API[Signal API]
        MODAL[Modal Runtime]
    end
    
    subgraph "Authentication & Database"
        SUPABASE[Supabase<br/>PostgreSQL + Auth]
    end
    
    subgraph "GPU Infrastructure"
        MODAL_GPU[Modal GPU<br/>L40S, A100, H100]
    end
    
    subgraph "Storage"
        S3[AWS S3<br/>Checkpoints]
        HF[HuggingFace Hub<br/>Models & Artifacts]
    end
    
    subgraph "Billing & Credits"
        FRONTIER[Frontier Backend<br/>Credit Management]
    end
    
    subgraph "Monitoring & Logging"
        WANDB[Weights & Biases<br/>Experiment Tracking]
        SECURITY[Security Logs<br/>Audit Trail]
    end
    
    API --> SUPABASE
    API --> FRONTIER
    API --> MODAL
    
    MODAL --> MODAL_GPU
    MODAL --> S3
    MODAL --> HF
    MODAL --> WANDB
    
    API --> SECURITY
```

This visualization shows Signal as a sophisticated ML training platform that abstracts away infrastructure complexity while providing fine-grained control over the training process. The architecture is designed for scalability, reliability, and ease of use.
