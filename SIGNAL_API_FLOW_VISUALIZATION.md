# Signal API Complete Flow Visualization

## High-Level Architecture Flow

```mermaid
graph TB
    subgraph "User Environment"
        A[User runs simple_example.py] --> B[SignalClient initialization]
        B --> C[Create Run Request]
    end
    
    subgraph "API Server (FastAPI)"
        C --> D[Authentication & Rate Limiting]
        D --> E[Model Validation & GPU Allocation]
        E --> F[Balance Check via Frontier Backend]
        F --> G[Run Registry (Supabase)]
        G --> H[Modal Container Lookup]
    end
    
    subgraph "Modal Infrastructure"
        H --> I[TrainingSession Container]
        I --> J[Model Loading & Initialization]
        J --> K[Ready for Training]
    end
    
    subgraph "Training Loop"
        K --> L[Forward-Backward Pass]
        L --> M[Optimizer Step]
        M --> N[Sample Generation]
        N --> O[Save State]
        O --> P{More Steps?}
        P -->|Yes| L
        P -->|No| Q[Training Complete]
    end
    
    subgraph "External Services"
        R[Supabase Database]
        S[Frontier Backend]
        T[S3/R2 Storage]
        U[HuggingFace Hub]
    end
    
    G -.-> R
    F -.-> S
    O -.-> T
    O -.-> U
```

## Detailed Request Flow

```mermaid
sequenceDiagram
    participant U as User Script
    participant C as SignalClient
    participant A as API Server
    participant S as Supabase
    participant F as Frontier Backend
    participant M as Modal Container
    participant HF as HuggingFace Hub
    participant S3 as S3 Storage
    
    Note over U,S3: Phase 1: Client Initialization
    U->>C: SignalClient(api_key)
    C->>C: Setup session & headers
    
    Note over U,S3: Phase 2: Run Creation
    U->>C: create_run(base_model="Llama-3.2-3B")
    C->>A: POST /runs (with config)
    A->>A: verify_auth() - validate API key
    A->>S: Query api_keys table
    S-->>A: Return user_id
    A->>A: Model validation & GPU allocation
    A->>F: GET /internal/get-balance
    F-->>A: Return balance
    A->>S: create_run_if_allowed()
    S-->>A: Return run_id
    A->>M: session.initialize.remote()
    
    Note over U,S3: Phase 3: Modal Container Startup
    M->>HF: Download Llama-3.2-3B model
    HF-->>M: Model files
    M->>M: Load model with LoRA adapters
    M->>M: Setup optimizer & accelerator
    M-->>A: Initialization complete
    A-->>C: RunResponse with run_id
    C-->>U: SignalRun object
    
    Note over U,S3: Phase 4: Training Loop (20 iterations)
    loop Training Steps
        U->>C: run.forward_backward(batch)
        C->>A: POST /runs/{id}/forward_backward
        A->>A: Check balance & charge
        A->>M: session.forward_backward.remote()
        M->>M: Tokenize batch & forward pass
        M->>M: Compute loss & backward pass
        M-->>A: Return loss & metrics
        A-->>C: ForwardBackwardResponse
        C-->>U: Loss & gradient norm
        
        U->>C: run.optim_step()
        C->>A: POST /runs/{id}/optim_step
        A->>M: session.optim_step.remote()
        M->>M: Apply optimizer update
        M-->>A: Return step & learning rate
        A-->>C: OptimStepResponse
        C-->>U: Step number
        
        alt Every 5 steps
            U->>C: run.sample(prompts)
            C->>A: POST /runs/{id}/sample
            A->>M: session.sample.remote()
            M->>M: Generate text completions
            M-->>A: Return generated text
            A-->>C: SampleResponse
            C-->>U: Generated completions
        end
    end
    
    Note over U,S3: Phase 5: Model Saving
    U->>C: run.save_state(mode="adapter")
    C->>A: POST /runs/{id}/save_state
    A->>M: session.save_state.remote()
    M->>M: Save LoRA adapters
    M->>S3: Upload checkpoint
    S3-->>M: Return S3 URI
    M-->>A: Return save paths & URLs
    A->>S: Record artifact metadata
    A-->>C: SaveStateResponse
    C-->>U: Save confirmation
```

## Component Interaction Map

```mermaid
graph LR
    subgraph "Client Layer"
        SC[SignalClient]
        SR[SignalRun]
        TC[TrainingClient]
        IC[InferenceClient]
    end
    
    subgraph "API Layer"
        FA[FastAPI Server]
        AM[AuthManager]
        RR[RunRegistry]
        MR[ModelRegistry]
        GA[GPUAllocator]
        FC[FrontierClient]
    end
    
    subgraph "Infrastructure"
        SB[(Supabase DB)]
        FB[Frontier Backend]
        MD[Modal Containers]
        S3[S3/R2 Storage]
        HF[HuggingFace Hub]
    end
    
    subgraph "Modal Runtime"
        TS[TrainingSession]
        ML[ModelLoader]
        GM[GPUMonitor]
        S3C[S3Client]
    end
    
    SC --> FA
    SR --> TC
    SR --> IC
    TC --> FA
    IC --> FA
    
    FA --> AM
    FA --> RR
    FA --> MR
    FA --> GA
    FA --> FC
    
    AM --> SB
    RR --> SB
    FC --> FB
    FA --> MD
    
    MD --> TS
    TS --> ML
    TS --> GM
    TS --> S3C
    
    ML --> HF
    S3C --> S3
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Input Data"
        A[Training Batch]
        B[Model Configuration]
        C[API Key]
    end
    
    subgraph "Processing Pipeline"
        D[Tokenization]
        E[Forward Pass]
        F[Loss Computation]
        G[Backward Pass]
        H[Gradient Accumulation]
        I[Optimizer Step]
    end
    
    subgraph "Output Data"
        J[Model Checkpoints]
        K[Generated Text]
        L[Training Metrics]
        M[Cost Information]
    end
    
    subgraph "Storage Systems"
        N[Modal Volume]
        O[S3/R2 Storage]
        P[Supabase Database]
        Q[HuggingFace Hub]
    end
    
    A --> D
    B --> E
    C --> E
    
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    
    I --> J
    E --> K
    F --> L
    I --> M
    
    J --> N
    J --> O
    L --> P
    M --> P
    J --> Q
```

## Performance Characteristics

```mermaid
gantt
    title Signal API Performance Timeline
    dateFormat X
    axisFormat %s
    
    section Cold Start
    Client Init           :0, 1
    API Auth             :1, 2
    Model Loading        :2, 62
    Container Ready      :62, 63
    
    section Warm Operations
    Forward-Backward     :63, 68
    Optimizer Step       :68, 69
    Sample Generation    :69, 79
    Save State           :79, 109
    
    section Training Loop
    Step 1-20            :109, 209
    Gradient Accumulation:209, 309
    Learning Rate Sched  :309, 409
```

## Error Handling Flow

```mermaid
graph TD
    A[Request Received] --> B{Authentication Valid?}
    B -->|No| C[401 Unauthorized]
    B -->|Yes| D{Rate Limit OK?}
    D -->|No| E[429 Rate Limited]
    D -->|Yes| F{Balance Sufficient?}
    F -->|No| G[402 Payment Required]
    F -->|Yes| H{Model Supported?}
    H -->|No| I[400 Bad Request]
    H -->|Yes| J{Modal Available?}
    J -->|No| K[500 Service Unavailable]
    J -->|Yes| L[Process Request]
    L --> M{Success?}
    M -->|No| N[500 Internal Error]
    M -->|Yes| O[200 Success Response]
    
    C --> P[Log Security Event]
    E --> Q[Log Rate Limit Event]
    G --> R[Log Billing Event]
    I --> S[Log Validation Error]
    K --> T[Log Infrastructure Error]
    N --> U[Log Processing Error]
    O --> V[Log Success Event]
```

This comprehensive visualization shows the complete flow of the Signal API from user script execution through all the infrastructure components, highlighting the key interactions, data flows, and performance characteristics.
