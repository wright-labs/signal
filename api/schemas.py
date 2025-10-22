"""Pydantic schemas for API requests and responses. I got pydantic/type checking pilled at my last job."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class RunConfig(BaseModel):
    """Configuration for creating a new training run."""

    base_model: str = Field(..., description="HuggingFace model ID or path")
    gpu_config: Optional[str] = Field(
        None,
        description="GPU configuration override (e.g., 'L40S:2', 'A100-80GB:4'). "
        "If not provided, automatically allocated based on model size.",
    )
    lora_r: int = Field(32, ge=1, le=512, description="LoRA rank (1-512)")
    lora_alpha: int = Field(
        32, ge=1, le=1024, description="LoRA alpha parameter (1-1024)"
    )
    lora_dropout: float = Field(
        0.0, ge=0.0, le=0.5, description="LoRA dropout rate (0.0-0.5)"
    )
    lora_target_modules: Optional[List[str]] = Field(
        None, description="Target modules for LoRA (None = auto)"
    )
    optimizer: str = Field("adamw_8bit", description="Optimizer type")
    learning_rate: float = Field(
        3e-4, gt=0, le=1.0, description="Learning rate (0-1.0)"
    )
    weight_decay: float = Field(
        0.01, ge=0.0, le=1.0, description="Weight decay (0.0-1.0)"
    )
    max_seq_length: int = Field(
        2048, ge=128, le=8192, description="Maximum sequence length (128-8192)" # TODO: maybe set a much higher max length?
    )
    bf16: bool = Field(True, description="Use bfloat16 precision")
    gradient_checkpointing: bool = Field(
        True, description="Enable gradient checkpointing"
    )

    @field_validator("gpu_config")
    @classmethod
    def validate_gpu_config(cls, v: Optional[str]) -> Optional[str]:
        """Validate GPU configuration format."""
        if v is not None:
            from api.gpu_allocator import validate_gpu_config as validate_gpu_config_func, GPUConfigError
            try:
                validate_gpu_config_func(v, raise_http_exception=False)
            except GPUConfigError as e:
                raise ValueError(str(e))
        return v

    @field_validator("base_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Prevent path traversal and validate model name format."""
        if ".." in v:
            raise ValueError("Model name cannot contain '..'")
        if "/" not in v:
            raise ValueError("Model name must be in format 'org/model-name'")
        parts = v.split("/")
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

# TODO: maybe set a much higher max length?
class TrainingExample(BaseModel):
    """Individual training example with validation.

    Supports multiple formats:
    1. SFT (Supervised Fine-Tuning): {"text": "..."} or {"messages": [...]}
    2. DPO (Direct Preference Optimization): {"prompt": "...", "chosen": "...", "rejected": "..."}
    3. GRPO (Group Relative Policy Optimization): {"prompt": "...", "responses": [...], "rewards": [...]}
    4. PPO (Proximal Policy Optimization): {"prompt": "...", "response": "...", "reward": 1.0, "value": 0.5}
    """

    # SFT formats
    text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=32768,
        description="Raw text for SFT (max 32K chars)",
    )
    messages: Optional[List[Dict[str, str]]] = Field(
        None, max_length=50, description="Chat messages for SFT (max 50)"
    )

    # DPO format
    prompt: Optional[str] = Field(
        None,
        min_length=1,
        max_length=32768,
        description="Prompt for preference-based methods",
    )
    chosen: Optional[str] = Field(
        None, min_length=1, max_length=32768, description="Chosen response for DPO"
    )
    rejected: Optional[str] = Field(
        None, min_length=1, max_length=32768, description="Rejected response for DPO"
    )

    # GRPO format
    responses: Optional[List[str]] = Field(
        None, max_length=16, description="Multiple responses for GRPO (max 16)"
    )
    rewards: Optional[List[float]] = Field(
        None, description="Reward scores for each response"
    )

    # PPO format
    response: Optional[str] = Field(
        None, max_length=32768, description="Single response for PPO"
    )
    reward: Optional[float] = Field(None, description="Reward score for PPO response")
    value: Optional[float] = Field(None, description="Value function estimate for PPO")

    @model_validator(mode="after")
    def validate_format(self) -> "TrainingExample":
        """Ensure exactly one valid format is provided."""
        has_text = self.text is not None
        has_messages = self.messages is not None

        # Check for each format
        has_dpo = all(
            [
                self.prompt is not None,
                self.chosen is not None,
                self.rejected is not None,
            ]
        )
        has_grpo = all(
            [
                self.prompt is not None,
                self.responses is not None,
                self.rewards is not None,
            ]
        )
        has_ppo = all(
            [
                self.prompt is not None,
                self.response is not None,
                self.reward is not None,
            ]
        )

        format_count = sum([has_text, has_messages, has_dpo, has_grpo, has_ppo])

        if format_count == 0:
            raise ValueError(
                "Must provide one of: 'text' (SFT), 'messages' (SFT), "
                "'(prompt, chosen, rejected)' (DPO), "
                "'(prompt, responses, rewards)' (GRPO), or "
                "'(prompt, response, reward)' (PPO)"
            )

        if format_count > 1:
            raise ValueError("Provide only ONE format: SFT, DPO, GRPO, or PPO")

        # TODO: wait why don't we just delete this section
        # Validate format-specific requirements
        if has_dpo:
            # Check if any DPO fields are partially set (shouldn't happen due to all() check above)
            pass  # Already validated by the all() check

        if has_grpo:
            if len(self.responses) != len(self.rewards):
                raise ValueError(
                    "Number of responses must match number of rewards for GRPO"
                )
        # TODO: wait why don't we just delete this section
        if has_ppo:
            # Check if PPO fields are partially set (shouldn't happen due to all() check above)
            pass  # Already validated by the all() check

        return self


class ForwardBackwardRequest(BaseModel):
    """Request for forward-backward pass."""

    batch_data: List[TrainingExample] = Field(
        ..., min_items=1, max_items=128, description="List of training examples (1-128)" # TODO: once again, should probably allow larger batch sizes?
    )
    accumulate: bool = Field(
        False, description="Accumulate gradients instead of replacing"
    )
    loss_fn: str = Field(
        "causal_lm",
        description="Loss function to use (causal_lm, dpo, grpo, ppo, enhanced_ppo, importance_sampling, conservative_ppo, reward_modeling)",
    )
    loss_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional arguments for loss function"
    )

    # Futures support
    request_id: Optional[str] = Field(
        None,
        description="Request ID for futures tracking (auto-generated if not provided)",
    )

    # RL-specific fields
    old_log_probs: Optional[List[float]] = Field(
        None, description="Old log probs from policy rollout (for PPO)"
    )
    rewards: Optional[List[float]] = Field(None, description="Rewards per example")
    values: Optional[List[float]] = Field(
        None, description="Value function estimates (for PPO)"
    )
    old_values: Optional[List[float]] = Field(
        None, description="Old value estimates for value clipping"
    )
    advantages: Optional[List[float]] = Field(
        None, description="Pre-computed advantages"
    )
    behavior_log_probs: Optional[List[float]] = Field(
        None, description="Behavior policy log probs (for importance sampling)"
    )

    # Reference model for KL penalty
    reference_model: Optional[str] = Field(
        None, description="Reference model name for KL divergence penalty"
    )

    # TODO: need to study up and see if I need this
    # GAE parameters
    use_gae: bool = Field(False, description="Use Generalized Advantage Estimation")
    gamma: float = Field(0.99, description="Discount factor for GAE")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")


class ForwardBackwardResponse(BaseModel):
    """Response from forward-backward pass."""

    loss: float
    step: int
    grad_norm: Optional[float] = None
    grad_stats: Optional[Dict[str, float]] = None

    # Futures support
    request_id: Optional[str] = Field(
        None, description="Request ID for futures tracking"
    )
    status: str = Field(
        "completed", description="Request status: queued, running, completed, failed"
    )

    # Comprehensive RL metrics
    rl_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="RL-specific metrics (policy_loss, value_loss, entropy, kl_divergence, etc.)",
    )
    metrics: Optional[Dict[str, float]] = Field(
        None, description="All training metrics"
    )


class OptimStepRequest(BaseModel):
    """Request for optimizer step."""
    # TODO: should probably make a learning rate scheduler?
    learning_rate: Optional[float] = Field(
        None, description="Override learning rate for this step"
    )


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
    token_ids: List[List[int]] = Field(
        default_factory=list, description="Token IDs for each generated output"
    )
    tokens: List[List[str]] = Field(
        default_factory=list, description="Token strings for each generated output"
    )
    logprobs: Optional[List[List[float]]] = None


class SaveStateRequest(BaseModel):
    """Request for saving model state."""

    mode: Literal["adapter", "merged"] = Field(
        "adapter", description="Save LoRA adapter or merged model"
    )
    push_to_hub: bool = Field(False, description="Push to HuggingFace Hub")
    hub_model_id: Optional[str] = Field(None, description="HuggingFace Hub model ID")


class SaveStateResponse(BaseModel):
    """Response from saving state."""

    artifact_uri: str  # Local path (backward compatibility) # TODO: do i need this anymore with R2?
    local_path: Optional[str] = None  # Explicit local path
    checkpoint_path: str  # Deprecated, use local_path # TODO: do i need this anymore with R2?
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

    text: str | List[str] = Field(
        ..., description="Text string or list of strings to tokenize"
    )
    add_special_tokens: bool = Field(
        True, description="Whether to add special tokens (BOS, EOS)"
    )


class TokenizeResponse(BaseModel):
    """Response from tokenization."""

    token_ids: List[List[int]] = Field(..., description="Token IDs for each input text")
    tokens: List[List[str]] = Field(
        ..., description="Token strings for each input text"
    )


class DetokenizeRequest(BaseModel):
    """Request for detokenizing token IDs."""

    token_ids: List[int] | List[List[int]] = Field(
        ..., description="Token IDs (single list or list of lists)"
    )


class DetokenizeResponse(BaseModel):
    """Response from detokenization."""

    text: str | List[str] = Field(
        ..., description="Decoded text (string or list of strings)"
    )


class TokenizerInfoResponse(BaseModel):
    """Tokenizer configuration information."""

    vocab_size: int = Field(..., description="Size of vocabulary")
    model_max_length: Optional[int] = Field(None, description="Maximum sequence length")
    bos_token_id: Optional[int] = Field(
        None, description="Beginning of sequence token ID"
    )
    eos_token_id: Optional[int] = Field(None, description="End of sequence token ID")
    pad_token_id: Optional[int] = Field(None, description="Padding token ID")
    unk_token_id: Optional[int] = Field(None, description="Unknown token ID")
    special_tokens: Dict[str, str] = Field(
        default_factory=dict, description="Special token strings"
    )


class ModelInfoResponse(BaseModel):
    """Model architecture information."""

    base_model: str = Field(..., description="Base model name")
    architecture: str = Field(..., description="Model architecture type")
    num_parameters: int = Field(..., description="Total number of parameters")
    num_trainable_parameters: int = Field(
        ..., description="Number of trainable parameters"
    )
    hidden_size: Optional[int] = Field(None, description="Hidden layer size")
    num_layers: Optional[int] = Field(None, description="Number of transformer layers")
    num_attention_heads: Optional[int] = Field(
        None, description="Number of attention heads"
    )
    vocab_size: Optional[int] = Field(
        None, description="Vocabulary size from model config"
    )
    max_position_embeddings: Optional[int] = Field(
        None, description="Maximum position embeddings"
    )
    chat_template: Optional[str] = Field(None, description="Chat template if available")

# TODO: once again should decide if i need chat template anymore
class ApplyChatTemplateRequest(BaseModel):
    """Request for applying chat template."""

    messages: List[Dict[str, str]] = Field(
        ..., description="List of message dicts with 'role' and 'content'"
    )
    add_generation_prompt: bool = Field(
        False, description="Whether to add generation prompt at the end"
    )

# TODO: once again should decide if i need chat template anymore
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

    texts: List[str] = Field(
        ..., min_items=1, max_items=128, description="List of texts to embed (1-128)"
    )
    layer: int = Field(
        -1, description="Layer to extract embeddings from (-1 for last layer)"
    )
    pooling: Literal["mean", "last_token", "cls_token"] = Field(
        "mean", description="Pooling strategy for embeddings"
    )


class EmbeddingsResponse(BaseModel):
    """Response from embeddings generation."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimensions: int = Field(..., description="Dimensionality of embeddings")


class RequestStatusResponse(BaseModel):
    """Response for checking request status (futures support)."""

    request_id: str = Field(..., description="Request ID")
    run_id: str = Field(..., description="Run ID")
    request_type: str = Field(
        ..., description="Type of request (forward_backward, optim_step, etc.)"
    )
    status: str = Field(
        ..., description="Request status: queued, running, completed, failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        None, description="Result data (if completed)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")
    submitted_at: Optional[float] = Field(None, description="Submission timestamp")
    started_at: Optional[float] = Field(None, description="Start timestamp")
    completed_at: Optional[float] = Field(None, description="Completion timestamp")


class EvaluateRequest(BaseModel):
    """Request for policy evaluation."""

    eval_prompts: List[str] = Field(
        ..., min_items=1, max_items=100, description="Evaluation prompts (1-100)"
    )
    reference_model: Optional[str] = Field(
        None, description="Reference model for KL divergence"
    )
    max_tokens: int = Field(100, description="Maximum tokens to generate per prompt")
    num_samples_per_prompt: int = Field(
        5, ge=1, le=20, description="Number of samples per prompt (1-20)"
    )
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")


class EvaluateResponse(BaseModel):
    """Response from policy evaluation."""

    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    kl_divergence: Optional[float] = Field(
        None, description="KL divergence from reference"
    )
    perplexity: float = Field(..., description="Model perplexity")
    entropy: float = Field(..., description="Policy entropy")
    unique_token_ratio: float = Field(..., description="Unique token ratio (diversity)")
    unique_bigram_ratio: float = Field(..., description="Unique bigram ratio")
    unique_trigram_ratio: float = Field(..., description="Unique trigram ratio")
    avg_length: float = Field(..., description="Average generation length")
    num_samples: int = Field(..., description="Total number of samples evaluated")


class QueueStatsResponse(BaseModel):
    """Response with queue statistics."""

    run_id: str = Field(..., description="Run ID")
    total: int = Field(..., description="Total requests in queue")
    queued: int = Field(..., description="Queued requests")
    running: int = Field(..., description="Running requests")
    completed: int = Field(..., description="Completed requests")
    failed: int = Field(..., description="Failed requests")
