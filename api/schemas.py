"""Pydantic schemas for API requests and responses."""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class RunConfig(BaseModel):
    """Configuration for creating a new training run."""
    base_model: str = Field(..., description="HuggingFace model ID or path")
    gpu_config: Optional[str] = Field(None, description="GPU configuration (e.g., 'l40s:1', 'a100-80gb:2'). If not provided, uses model's default.")
    lora_r: int = Field(32, ge=1, le=512, description="LoRA rank (1-512)")
    lora_alpha: int = Field(64, ge=1, le=1024, description="LoRA alpha parameter (1-1024)")
    lora_dropout: float = Field(0.0, ge=0.0, le=0.5, description="LoRA dropout rate (0.0-0.5)")
    lora_target_modules: Optional[List[str]] = Field(None, description="Target modules for LoRA (None = auto)")
    optimizer: str = Field("adamw_8bit", description="Optimizer type")
    learning_rate: float = Field(3e-4, gt=0, le=1.0, description="Learning rate (0-1.0)")
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="Weight decay (0.0-1.0)")
    max_seq_length: int = Field(2048, ge=128, le=8192, description="Maximum sequence length (128-8192)")
    bf16: bool = Field(True, description="Use bfloat16 precision")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    
    @field_validator('gpu_config')
    @classmethod
    def validate_gpu_config(cls, v: Optional[str]) -> Optional[str]:
        """Validate GPU configuration format."""
        if v is None:
            return v

        valid_gpus = ["l40s", "a100-80gb", "a100", "h100", "t4", "a10g"]
        
        if ':' not in v:
            raise ValueError("GPU config must be in format 'gpu_type:count' (e.g., 'l40s:1')")
        
        gpu_type, count_str = v.rsplit(':', 1)
        
        if gpu_type not in valid_gpus:
            raise ValueError(f"GPU type '{gpu_type}' not supported. Valid types: {', '.join(valid_gpus)}")
        
        try:
            count = int(count_str)
            if count < 1 or count > 8:
                raise ValueError("GPU count must be between 1 and 8")
        except ValueError:
            raise ValueError("GPU count must be a valid integer")
        
        return v
    
    @field_validator('base_model')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Prevent path traversal and validate model name format."""
        if '..' in v:
            raise ValueError("Model name cannot contain '..'")
        if '/' not in v:
            raise ValueError("Model name must be in format 'org/model-name'")
        parts = v.split('/')
        if len(parts) < 2 or not all(parts):
            raise ValueError("Invalid model name format")
        return v


class RunResponse(BaseModel):
    """Response when creating a new run."""
    run_id: str
    user_id: str
    base_model: str
    status: str
    created_at: str
    config: Dict[str, Any]


class TrainingExample(BaseModel):
    """Individual training example with validation."""
    text: Optional[str] = Field(None, max_length=32768, description="Raw text (max 32K chars)")
    messages: Optional[List[Dict[str, str]]] = Field(None, max_length=50, description="Chat messages (max 50)")
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: Optional[str]) -> Optional[str]:
        """Ensure text is not empty if provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Text cannot be empty or whitespace only")
        return v
    
    @field_validator('messages')
    @classmethod
    def validate_messages_format(cls, v: Optional[List[Dict[str, str]]]) -> Optional[List[Dict[str, str]]]:
        """Validate message format."""
        if v is not None:
            for msg in v:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError("Each message must have 'role' and 'content' keys")
                if msg['role'] not in ['system', 'user', 'assistant']:
                    raise ValueError("Message role must be 'system', 'user', or 'assistant'")
        return v
    
    @field_validator('messages')
    @classmethod
    def validate_at_least_one(cls, v: Optional[List[Dict[str, str]]], info) -> Optional[List[Dict[str, str]]]:
        """Ensure at least one of text or messages is provided."""
        # Access other field values via info.data
        if v is None and info.data.get('text') is None:
            raise ValueError("Either 'text' or 'messages' must be provided")
        return v


class ForwardBackwardRequest(BaseModel):
    """Request for forward-backward pass."""
    batch_data: List[TrainingExample] = Field(
        ..., 
        min_items=1, 
        max_items=128, 
        description="List of training examples (1-128)"
    )
    accumulate: bool = Field(False, description="Accumulate gradients instead of replacing")
    loss_fn: str = Field("causal_lm", description="Loss function to use (causal_lm, dpo, reward_modeling, ppo)")
    loss_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional arguments for loss function")


class ForwardBackwardResponse(BaseModel):
    """Response from forward-backward pass."""
    loss: float
    step: int
    grad_norm: Optional[float] = None
    grad_stats: Optional[Dict[str, float]] = None


class OptimStepRequest(BaseModel):
    """Request for optimizer step."""
    learning_rate: Optional[float] = Field(None, description="Override learning rate for this step")


class OptimStepResponse(BaseModel):
    """Response from optimizer step."""
    step: int
    learning_rate: float
    metrics: Dict[str, Any]


class SampleRequest(BaseModel):
    """Request for sampling/generation."""
    prompts: List[str] = Field(..., description="List of prompts to generate from")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    return_logprobs: bool = Field(False, description="Return log probabilities")


class SampleResponse(BaseModel):
    """Response from sampling."""
    outputs: List[str]
    token_ids: List[List[int]] = Field(default_factory=list, description="Token IDs for each generated output")
    tokens: List[List[str]] = Field(default_factory=list, description="Token strings for each generated output")
    logprobs: Optional[List[List[float]]] = None


class SaveStateRequest(BaseModel):
    """Request for saving model state."""
    mode: Literal["adapter", "merged"] = Field("adapter", description="Save LoRA adapter or merged model")
    push_to_hub: bool = Field(False, description="Push to HuggingFace Hub")
    hub_model_id: Optional[str] = Field(None, description="HuggingFace Hub model ID")


class SaveStateResponse(BaseModel):
    """Response from saving state."""
    artifact_uri: str  # Local path (backward compatibility)
    local_path: Optional[str] = None  # Explicit local path
    checkpoint_path: str  # Deprecated, use local_path
    s3_uri: Optional[str] = None  # S3 URI for permanent storage
    download_url: Optional[str] = None  # Pre-signed S3 download URL
    download_expires_at: Optional[str] = None  # Expiration timestamp for download URL
    manifest: Optional[Dict[str, Any]] = None  # Full manifest metadata
    pushed_to_hub: bool
    hub_model_id: Optional[str] = None


class RunStatus(BaseModel):
    """Status information for a run."""
    run_id: str
    user_id: str
    base_model: str
    status: str
    current_step: int
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    cost_so_far: Optional[float] = 0.0
    charged_so_far: Optional[float] = 0.0
    cost_per_hour: Optional[float] = 0.0


class RunMetrics(BaseModel):
    """Metrics for a run."""
    run_id: str
    step: int
    metrics: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class TokenizeRequest(BaseModel):
    """Request for tokenizing text."""
    text: str | List[str] = Field(..., description="Text string or list of strings to tokenize")
    add_special_tokens: bool = Field(True, description="Whether to add special tokens (BOS, EOS)")


class TokenizeResponse(BaseModel):
    """Response from tokenization."""
    token_ids: List[List[int]] = Field(..., description="Token IDs for each input text")
    tokens: List[List[str]] = Field(..., description="Token strings for each input text")


class DetokenizeRequest(BaseModel):
    """Request for detokenizing token IDs."""
    token_ids: List[int] | List[List[int]] = Field(..., description="Token IDs (single list or list of lists)")


class DetokenizeResponse(BaseModel):
    """Response from detokenization."""
    text: str | List[str] = Field(..., description="Decoded text (string or list of strings)")


class TokenizerInfoResponse(BaseModel):
    """Tokenizer configuration information."""
    vocab_size: int = Field(..., description="Size of vocabulary")
    model_max_length: Optional[int] = Field(None, description="Maximum sequence length")
    bos_token_id: Optional[int] = Field(None, description="Beginning of sequence token ID")
    eos_token_id: Optional[int] = Field(None, description="End of sequence token ID")
    pad_token_id: Optional[int] = Field(None, description="Padding token ID")
    unk_token_id: Optional[int] = Field(None, description="Unknown token ID")
    special_tokens: Dict[str, str] = Field(default_factory=dict, description="Special token strings")


class ModelInfoResponse(BaseModel):
    """Model architecture information."""
    base_model: str = Field(..., description="Base model name")
    architecture: str = Field(..., description="Model architecture type")
    num_parameters: int = Field(..., description="Total number of parameters")
    num_trainable_parameters: int = Field(..., description="Number of trainable parameters")
    hidden_size: Optional[int] = Field(None, description="Hidden layer size")
    num_layers: Optional[int] = Field(None, description="Number of transformer layers")
    num_attention_heads: Optional[int] = Field(None, description="Number of attention heads")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size from model config")
    max_position_embeddings: Optional[int] = Field(None, description="Maximum position embeddings")
    chat_template: Optional[str] = Field(None, description="Chat template if available")


class ApplyChatTemplateRequest(BaseModel):
    """Request for applying chat template."""
    messages: List[Dict[str, str]] = Field(..., description="List of message dicts with 'role' and 'content'")
    add_generation_prompt: bool = Field(False, description="Whether to add generation prompt at the end")


class ApplyChatTemplateResponse(BaseModel):
    """Response from applying chat template."""
    text: str = Field(..., description="Formatted text with chat template applied")
    token_ids: List[int] = Field(..., description="Token IDs of formatted text")


class StreamSampleRequest(BaseModel):
    """Request for streaming text generation."""
    prompt: str = Field(..., description="Single prompt to generate from")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")


class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""
    token: str = Field(..., description="Generated token text")
    token_id: int = Field(..., description="Token ID")
    logprob: Optional[float] = Field(None, description="Log probability of this token")
    is_finished: bool = Field(False, description="Whether generation is complete")


class EmbeddingsRequest(BaseModel):
    """Request for generating embeddings."""
    texts: List[str] = Field(..., min_items=1, max_items=128, description="List of texts to embed (1-128)")
    layer: int = Field(-1, description="Layer to extract embeddings from (-1 for last layer)")
    pooling: Literal["mean", "last_token", "cls_token"] = Field("mean", description="Pooling strategy for embeddings")


class EmbeddingsResponse(BaseModel):
    """Response from embeddings generation."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimensions: int = Field(..., description="Dimensionality of embeddings")