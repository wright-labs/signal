"""Pydantic schemas for Signal SDK requests and responses."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class RunConfig(BaseModel):
    """Configuration for creating a new training run."""
    
    base_model: str = Field(..., description="HuggingFace model ID or path")
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
        if v is None and info.data.get('text') is None:
            raise ValueError("Either 'text' or 'messages' must be provided")
        return v


class ForwardBackwardRequest(BaseModel):
    """Request for forward-backward pass."""
    
    batch_data: List[TrainingExample] = Field(
        ..., 
        min_length=1, 
        max_length=128, 
        description="List of training examples (1-128)"
    )
    accumulate: bool = Field(False, description="Accumulate gradients instead of replacing")


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
    logprobs: Optional[List[List[float]]] = None


class SaveStateRequest(BaseModel):
    """Request for saving model state."""
    
    mode: Literal["adapter", "merged"] = Field("adapter", description="Save LoRA adapter or merged model")
    push_to_hub: bool = Field(False, description="Push to HuggingFace Hub")
    hub_model_id: Optional[str] = Field(None, description="HuggingFace Hub model ID")


class SaveStateResponse(BaseModel):
    """Response from saving state."""
    
    artifact_uri: str
    checkpoint_path: str
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


class RunMetrics(BaseModel):
    """Metrics for a run."""
    
    run_id: str
    step: int
    metrics: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response from API."""
    
    error: str
    detail: Optional[str] = None
