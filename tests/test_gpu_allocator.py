"""Tests for automatic GPU allocation."""
import pytest
from api.gpu_allocator import estimate_model_parameters, allocate_gpu_config, validate_gpu_config


class TestEstimateModelParameters:
    """Test parameter estimation from model names."""
    
    def test_parse_standard_format(self):
        """Test parsing standard model name formats."""
        assert estimate_model_parameters("meta-llama/Llama-3.1-8B") == 8.0
        assert estimate_model_parameters("Qwen/Qwen2.5-7B") == 7.0
        assert estimate_model_parameters("meta-llama/Llama-3.1-70B") == 70.0
        assert estimate_model_parameters("Qwen/Qwen2.5-1.5B") == 1.5
        assert estimate_model_parameters("google/gemma-2-2b") == 2.0
    
    def test_parse_lowercase(self):
        """Test parsing lowercase 'b' variants."""
        assert estimate_model_parameters("model-7b") == 7.0
        assert estimate_model_parameters("model-1.5b") == 1.5
    
    def test_unknown_format(self):
        """Test handling unknown model formats."""
        # Should return None for models without size in name
        result = estimate_model_parameters("unknown/model-name")
        assert result is None


class TestAllocateGPUConfig:
    """Test GPU configuration allocation logic."""
    
    def test_sub_1b_models(self):
        """Test < 1B parameter models get L40S:1."""
        assert allocate_gpu_config("test/model-0.5B", parameter_count=0.5) == "L40S:1"
    
    def test_1b_to_3b_models(self):
        """Test 1B-3B parameter models get L40S:1."""
        assert allocate_gpu_config("test/model-1B", parameter_count=1.0) == "L40S:1"
        assert allocate_gpu_config("test/model-2B", parameter_count=2.0) == "L40S:1"
    
    def test_3b_to_7b_models(self):
        """Test 3B-7B parameter models get L40S:1."""
        assert allocate_gpu_config("test/model-3B", parameter_count=3.0) == "L40S:1"
        assert allocate_gpu_config("test/model-6B", parameter_count=6.9) == "L40S:1"
    
    def test_7b_to_13b_models(self):
        """Test 7B-13B parameter models get A100-80GB:1."""
        assert allocate_gpu_config("Qwen/Qwen2.5-7B", parameter_count=7.0) == "A100-80GB:1"
        assert allocate_gpu_config("test/model-8B", parameter_count=8.0) == "A100-80GB:1"
        assert allocate_gpu_config("test/model-12B", parameter_count=12.9) == "A100-80GB:1"
    
    def test_13b_to_30b_models(self):
        """Test 13B-30B parameter models get A100-80GB:2."""
        assert allocate_gpu_config("test/model-13B", parameter_count=13.0) == "A100-80GB:2"
        assert allocate_gpu_config("test/model-14B", parameter_count=14.0) == "A100-80GB:2"
        assert allocate_gpu_config("test/model-29B", parameter_count=29.9) == "A100-80GB:2"
    
    def test_30b_to_70b_models(self):
        """Test 30B-70B parameter models get A100-80GB:4."""
        assert allocate_gpu_config("test/model-30B", parameter_count=30.0) == "A100-80GB:4"
        assert allocate_gpu_config("test/model-32B", parameter_count=32.0) == "A100-80GB:4"
        assert allocate_gpu_config("test/model-69B", parameter_count=69.9) == "A100-80GB:4"
    
    def test_over_70b_models(self):
        """Test > 70B parameter models get A100-80GB:8."""
        assert allocate_gpu_config("test/model-100B", parameter_count=100.0) == "A100-80GB:8"
    
    def test_user_override(self):
        """Test user override takes precedence."""
        assert allocate_gpu_config("test/model-8B", user_override="L40S:2", parameter_count=8.0) == "L40S:2"
        assert allocate_gpu_config("test/model-70B", user_override="H100:4", parameter_count=70.0) == "H100:4"
    
    def test_automatic_estimation(self):
        """Test automatic parameter estimation from model name."""
        # Should parse from name and allocate correctly
        config = allocate_gpu_config("Qwen/Qwen2.5-7B")
        assert config == "A100-80GB:1"  # 7B gets A100-80GB:1
        
        config = allocate_gpu_config("meta-llama/Llama-3.1-70B")
        assert config == "A100-80GB:8"  # 70B gets A100-80GB:8
    
    def test_unknown_model_conservative_default(self):
        """Test unknown models get conservative allocation."""
        # Should default to A100-80GB:2 for safety
        config = allocate_gpu_config("unknown/model-without-size")
        assert config == "A100-80GB:2"


class TestValidateGPUConfig:
    """Test GPU configuration validation."""
    
    def test_valid_configs(self):
        """Test valid GPU configurations."""
        assert validate_gpu_config("L40S:1") == True
        assert validate_gpu_config("A100-80GB:4") == True
        assert validate_gpu_config("H100:2") == True
        assert validate_gpu_config("T4:1") == True
    
    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="must be in format"):
            validate_gpu_config("L40S")
        
        with pytest.raises(ValueError, match="must be in format"):
            validate_gpu_config("invalid")
    
    def test_invalid_gpu_type(self):
        """Test invalid GPU type raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            validate_gpu_config("INVALID:1")
        
        with pytest.raises(ValueError, match="not supported"):
            validate_gpu_config("V100:1")  # Not in supported list
    
    def test_invalid_count(self):
        """Test invalid GPU count raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 8"):
            validate_gpu_config("L40S:0")
        
        with pytest.raises(ValueError, match="must be between 1 and 8"):
            validate_gpu_config("L40S:10")
        
        with pytest.raises(ValueError, match="must be a valid integer"):
            validate_gpu_config("L40S:abc")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

