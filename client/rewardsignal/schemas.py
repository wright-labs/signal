"""Pydantic schemas for Signal SDK requests and responses."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


class RunConfig(BaseModel):
    """Configuration for creating a new training run."""
    
    base_model: str = Field(..., description="HuggingFace model ID or path")
    gpu_config: Optional[str] = Field(
        None,
        description="GPU configuration override (e.g., 'L40S:2', 'A100-80GB:4'). "
                    "If not provided, automatically allocated based on model size."
    )
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
    """Individual training example with validation.
    
    Supports three formats:
    1. SFT (Supervised Fine-Tuning) with raw text: {"text": "..."}
    2. SFT with chat messages: {"messages": [...]}
    3. DPO (Direct Preference Optimization): {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    # SFT formats
    text: Optional[str] = Field(None, max_length=32768, description="Raw text for SFT (max 32K chars)")
    messages: Optional[List[Dict[str, str]]] = Field(None, max_length=50, description="Chat messages for SFT (max 50)")
    
    # DPO format
    prompt: Optional[str] = Field(None, max_length=32768, description="Prompt for preference-based methods")
    chosen: Optional[str] = Field(None, max_length=32768, description="Chosen response for DPO")
    rejected: Optional[str] = Field(None, max_length=32768, description="Rejected response for DPO")

    # GRPO format
    responses: Optional[List[str]] = Field(None, max_length=16, description="Multiple responses for GRPO (max 16)")
    rewards: Optional[List[float]] = Field(None, description="Reward scores for each response")

    # PPO format
    response: Optional[str] = Field(None, max_length=32768, description="Single response for PPO")
    reward: Optional[float] = Field(None, description="Reward score for PPO response")
    value: Optional[float] = Field(None, description="Value function estimate for PPO")
    
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
    
    @field_validator('prompt', 'chosen', 'rejected')
    @classmethod
    def validate_preference_not_empty(cls, v: Optional[str]) -> Optional[str]:
        """Ensure preference fields are not empty if provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Preference field cannot be empty or whitespace only")
        return v
    
    @model_validator(mode='after')
    def validate_format(self) -> 'TrainingExample':
        """Ensure exactly one valid format is provided."""
        has_text = self.text is not None
        has_messages = self.messages is not None

        # Check for each format
        has_dpo = all([
            self.prompt is not None,
            self.chosen is not None,
            self.rejected is not None
        ])
        has_grpo = all([
            self.prompt is not None,
            self.responses is not None,
            self.rewards is not None
        ])
        has_ppo = all([
            self.prompt is not None,
            self.response is not None,
            self.reward is not None
        ])

        format_count = sum([has_text, has_messages, has_dpo, has_grpo, has_ppo])

        if format_count == 0:
            raise ValueError(
                "Must provide one of: 'text' (SFT), 'messages' (SFT), "
                "'(prompt, chosen, rejected)' (DPO), "
                "'(prompt, responses, rewards)' (GRPO), or "
                "'(prompt, response, reward)' (PPO)"
            )

        if format_count > 1:
            raise ValueError(
                "Provide only ONE format: SFT, DPO, GRPO, or PPO"
            )

        return self


class ForwardBackwardRequest(BaseModel):
    """Request for forward-backward pass."""
    
    batch_data: List[TrainingExample] = Field(
        ..., 
        min_length=1, 
        max_length=128, 
        description="List of training examples (1-128)"
    )
    accumulate: bool = Field(False, description="Accumulate gradients instead of replacing")
    loss_fn: str = Field("causal_lm", description="Loss function to use (causal_lm, dpo, grpo, ppo)")
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
