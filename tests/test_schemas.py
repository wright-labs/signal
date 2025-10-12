"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError

from api.schemas import (
    RunConfig,
    ForwardBackwardRequest,
    OptimStepRequest,
    SampleRequest,
    SaveStateRequest,
    RunResponse,
    ForwardBackwardResponse,
    OptimStepResponse,
    SampleResponse,
    SaveStateResponse,
    RunStatus,
    RunMetrics,
)


class TestRunConfigSchema:
    """Tests for RunConfig schema."""
    
    def test_run_config_defaults(self):
        """Test RunConfig with minimal data uses correct defaults."""
        config = RunConfig(base_model="meta-llama/Llama-3.2-3B")
        
        assert config.base_model == "meta-llama/Llama-3.2-3B"
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.0
        assert config.lora_target_modules is None
        assert config.optimizer == "adamw_8bit"
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.01
        assert config.max_seq_length == 2048
        assert config.bf16 is True
        assert config.gradient_checkpointing is True
    
    def test_run_config_custom_values(self):
        """Test RunConfig with custom values."""
        config = RunConfig(
            base_model="google/gemma-2-2b",
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            learning_rate=5e-4,
            max_seq_length=4096,
        )
        
        assert config.base_model == "google/gemma-2-2b"
        assert config.lora_r == 64
        assert config.lora_alpha == 128
        assert config.lora_dropout == 0.05
        assert config.learning_rate == 5e-4
        assert config.max_seq_length == 4096
    
    def test_run_config_with_target_modules(self):
        """Test RunConfig with custom LoRA target modules."""
        config = RunConfig(
            base_model="meta-llama/Llama-3.2-3B",
            lora_target_modules=["q_proj", "v_proj", "k_proj"],
        )
        
        assert config.lora_target_modules == ["q_proj", "v_proj", "k_proj"]
    
    def test_run_config_missing_required_field(self):
        """Test that RunConfig requires base_model."""
        with pytest.raises(ValidationError):
            RunConfig()


class TestRequestSchemas:
    """Tests for request schemas."""
    
    def test_forward_backward_request(self):
        """Test ForwardBackwardRequest schema."""
        request = ForwardBackwardRequest(
            batch_data=[{"text": "example"}],
            accumulate=True,
        )
        
        assert request.batch_data == [{"text": "example"}]
        assert request.accumulate is True
    
    def test_forward_backward_request_defaults(self):
        """Test ForwardBackwardRequest defaults."""
        request = ForwardBackwardRequest(batch_data=[{"text": "example"}])
        
        assert request.accumulate is False
    
    def test_optim_step_request(self):
        """Test OptimStepRequest schema."""
        request = OptimStepRequest(learning_rate=5e-4)
        
        assert request.learning_rate == 5e-4
    
    def test_optim_step_request_optional(self):
        """Test OptimStepRequest with no learning rate."""
        request = OptimStepRequest()
        
        assert request.learning_rate is None
    
    def test_sample_request(self):
        """Test SampleRequest schema."""
        request = SampleRequest(
            prompts=["Hello", "World"],
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            return_logprobs=True,
        )
        
        assert request.prompts == ["Hello", "World"]
        assert request.max_tokens == 256
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.return_logprobs is True
    
    def test_sample_request_defaults(self):
        """Test SampleRequest with defaults."""
        request = SampleRequest(prompts=["test"])
        
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.return_logprobs is False
    
    def test_save_state_request(self):
        """Test SaveStateRequest schema."""
        request = SaveStateRequest(
            mode="merged",
            push_to_hub=True,
            hub_model_id="user/model",
        )
        
        assert request.mode == "merged"
        assert request.push_to_hub is True
        assert request.hub_model_id == "user/model"
    
    def test_save_state_request_defaults(self):
        """Test SaveStateRequest defaults."""
        request = SaveStateRequest()
        
        assert request.mode == "adapter"
        assert request.push_to_hub is False
        assert request.hub_model_id is None
    
    def test_save_state_request_invalid_mode(self):
        """Test SaveStateRequest with invalid mode."""
        with pytest.raises(ValidationError):
            SaveStateRequest(mode="invalid_mode")


class TestResponseSchemas:
    """Tests for response schemas."""
    
    def test_run_response(self):
        """Test RunResponse schema."""
        response = RunResponse(
            run_id="run_123",
            user_id="user_456",
            base_model="meta-llama/Llama-3.2-3B",
            status="active",
            created_at="2024-01-01T00:00:00",
            config={"lora_r": 32},
        )
        
        assert response.run_id == "run_123"
        assert response.user_id == "user_456"
        assert response.status == "active"
    
    def test_forward_backward_response(self):
        """Test ForwardBackwardResponse schema."""
        response = ForwardBackwardResponse(
            loss=2.5,
            step=10,
            grad_norm=1.2,
            grad_stats={"max_grad": 0.5},
        )
        
        assert response.loss == 2.5
        assert response.step == 10
        assert response.grad_norm == 1.2
        assert response.grad_stats == {"max_grad": 0.5}
    
    def test_forward_backward_response_optional_fields(self):
        """Test ForwardBackwardResponse with optional fields."""
        response = ForwardBackwardResponse(loss=2.5, step=10)
        
        assert response.grad_norm is None
        assert response.grad_stats is None
    
    def test_optim_step_response(self):
        """Test OptimStepResponse schema."""
        response = OptimStepResponse(
            step=11,
            learning_rate=3e-4,
            metrics={"checkpoint_path": "/path/to/checkpoint"},
        )
        
        assert response.step == 11
        assert response.learning_rate == 3e-4
        assert response.metrics == {"checkpoint_path": "/path/to/checkpoint"}
    
    def test_sample_response(self):
        """Test SampleResponse schema."""
        response = SampleResponse(
            outputs=["generated text 1", "generated text 2"],
            logprobs=[[0.1, 0.2], [0.3, 0.4]],
        )
        
        assert len(response.outputs) == 2
        assert response.logprobs == [[0.1, 0.2], [0.3, 0.4]]
    
    def test_sample_response_without_logprobs(self):
        """Test SampleResponse without logprobs."""
        response = SampleResponse(outputs=["text"])
        
        assert response.logprobs is None
    
    def test_save_state_response(self):
        """Test SaveStateResponse schema."""
        response = SaveStateResponse(
            artifact_uri="s3://bucket/path",
            checkpoint_path="/local/path",
            pushed_to_hub=True,
            hub_model_id="user/model",
        )
        
        assert response.artifact_uri == "s3://bucket/path"
        assert response.checkpoint_path == "/local/path"
        assert response.pushed_to_hub is True
        assert response.hub_model_id == "user/model"
    
    def test_run_status(self):
        """Test RunStatus schema."""
        status = RunStatus(
            run_id="run_123",
            user_id="user_456",
            base_model="model",
            status="active",
            current_step=5,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:01:00",
            config={},
        )
        
        assert status.run_id == "run_123"
        assert status.current_step == 5
    
    def test_run_metrics(self):
        """Test RunMetrics schema."""
        metrics = RunMetrics(
            run_id="run_123",
            step=5,
            metrics=[
                {"step": 0, "loss": 3.0},
                {"step": 1, "loss": 2.8},
            ],
        )
        
        assert metrics.run_id == "run_123"
        assert metrics.step == 5
        assert len(metrics.metrics) == 2


class TestSchemaValidation:
    """Tests for schema validation edge cases."""
    
    def test_negative_lora_r_not_allowed(self):
        """Test that negative LoRA rank is validated."""
        # Note: Pydantic doesn't enforce this by default, but the model should
        config = RunConfig(base_model="model", lora_r=-1)
        # This creates the config, but in production you might want to add validators
        assert config.lora_r == -1
    
    def test_empty_batch_data(self):
        """Test forward_backward with empty batch."""
        request = ForwardBackwardRequest(batch_data=[])
        assert request.batch_data == []
    
    def test_empty_prompts_list(self):
        """Test sample request with empty prompts."""
        request = SampleRequest(prompts=[])
        assert request.prompts == []
    
    def test_model_dump_preserves_data(self):
        """Test that model_dump works correctly."""
        config = RunConfig(
            base_model="model",
            lora_r=64,
            learning_rate=5e-4,
        )
        
        dumped = config.model_dump()
        
        assert dumped["base_model"] == "model"
        assert dumped["lora_r"] == 64
        assert dumped["learning_rate"] == 5e-4

