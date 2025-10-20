"""Tests for GRPO trainer integration."""
from unittest.mock import MagicMock, patch
from datasets import Dataset


def test_create_grpo_trainer():
    """Test GRPO trainer creation."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    # Mock model and tokenizer
    model = MagicMock()
    tokenizer = MagicMock()
    
    # Create simple dataset
    dataset = Dataset.from_dict({
        "prompt": ["What is 2+2?", "What is 3+3?"]
    })
    
    # Simple reward function
    def simple_reward(completions):
        """Reward based on whether '4' is in completion."""
        return [1.0 if "4" in c else 0.0 for c in completions]
    
    # Create trainer - patch the TRL import
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[simple_reward],
            train_dataset=dataset,
        )
        
        # Verify GRPOTrainer was called
        MockTrainer.assert_called_once()
        
        # Verify arguments
        call_kwargs = MockTrainer.call_args.kwargs
        assert call_kwargs['model'] == model
        assert call_kwargs['tokenizer'] == tokenizer
        assert len(call_kwargs['reward_funcs']) == 1


def test_create_grpo_trainer_with_dapo():
    """Test GRPO trainer with DAPO variant."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = Dataset.from_dict({"prompt": ["Test"]})
    
    def reward_fn(completions):
        return [1.0] * len(completions)
    
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=dataset,
            loss_type="dapo",
            epsilon=0.2,
            epsilon_high=0.28,
            mask_truncated_completions=True,
        )
        
        # Verify DAPO-specific config
        MockTrainer.assert_called_once()
        call_kwargs = MockTrainer.call_args.kwargs
        
        config = call_kwargs['config']
        assert config.loss_type == "dapo"
        assert config.epsilon == 0.2
        assert config.epsilon_high == 0.28
        assert config.mask_truncated_completions is True


def test_get_soft_overlong_punishment():
    """Test soft overlong punishment reward function."""
    from modal_runtime.trl_trainers import get_soft_overlong_punishment
    
    with patch('trl.get_soft_overlong_punishment') as mock_sop:
        mock_sop.return_value = lambda x: [0.0]
        
        reward_fn = get_soft_overlong_punishment(
            max_completion_len=20480,
            soft_punish_cache=4096,
        )
        
        # Verify TRL's function was called with correct args
        mock_sop.assert_called_once_with(
            max_completion_len=20480,
            soft_punish_cache=4096,
        )
        
        # Should return a callable
        assert callable(reward_fn)


def test_grpo_trainer_with_lite_ppo():
    """Test GRPO trainer with Lite PPO (batch-scaled rewards)."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = Dataset.from_dict({"prompt": ["Test"]})
    
    def reward_fn(completions):
        return [1.0] * len(completions)
    
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=dataset,
            scale_rewards="batch",  # Lite PPO
        )
        
        MockTrainer.assert_called_once()
        call_kwargs = MockTrainer.call_args.kwargs
        
        config = call_kwargs['config']
        assert config.scale_rewards == "batch"


def test_grpo_trainer_with_vllm():
    """Test GRPO trainer with vLLM and TIS correction."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = Dataset.from_dict({"prompt": ["Test"]})
    
    def reward_fn(completions):
        return [1.0] * len(completions)
    
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=dataset,
            use_vllm=True,
            vllm_importance_sampling_correction=True,
            vllm_importance_sampling_cap=2.0,
        )
        
        MockTrainer.assert_called_once()
        call_kwargs = MockTrainer.call_args.kwargs
        
        config = call_kwargs['config']
        assert config.use_vllm is True
        assert config.vllm_importance_sampling_correction is True
        assert config.vllm_importance_sampling_cap == 2.0


def test_multiple_reward_functions():
    """Test GRPO trainer with multiple reward functions."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = Dataset.from_dict({"prompt": ["Test"]})
    
    def length_reward(completions):
        """Reward by length."""
        return [len(c) * 0.01 for c in completions]
    
    def quality_reward(completions):
        """Reward by quality (dummy)."""
        return [1.0 if "good" in c.lower() else 0.0 for c in completions]
    
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[length_reward, quality_reward],
            train_dataset=dataset,
        )
        
        MockTrainer.assert_called_once()
        call_kwargs = MockTrainer.call_args.kwargs
        
        # Should have both reward functions
        assert len(call_kwargs['reward_funcs']) == 2


def test_grpo_config_parameters():
    """Test GRPO config parameters are properly set."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = Dataset.from_dict({"prompt": ["Test"]})
    
    def reward_fn(completions):
        return [1.0] * len(completions)
    
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=dataset,
            learning_rate=1e-5,
            beta=0.1,
            num_generations=16,
            max_prompt_length=512,
            max_completion_length=1024,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        )
        
        MockTrainer.assert_called_once()
        call_kwargs = MockTrainer.call_args.kwargs
        
        config = call_kwargs['config']
        assert config.learning_rate == 1e-5
        assert config.beta == 0.1
        assert config.num_generations == 16
        assert config.max_prompt_length == 512
        assert config.max_completion_length == 1024
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 4


def test_dr_grpo_variant():
    """Test Dr. GRPO variant."""
    from modal_runtime.trl_trainers import create_grpo_trainer
    
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = Dataset.from_dict({"prompt": ["Test"]})
    
    def reward_fn(completions):
        return [1.0] * len(completions)
    
    with patch('trl.GRPOTrainer') as MockTrainer:
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            train_dataset=dataset,
            loss_type="dr_grpo",
        )
        
        MockTrainer.assert_called_once()
        call_kwargs = MockTrainer.call_args.kwargs
        
        config = call_kwargs['config']
        assert config.loss_type == "dr_grpo"

