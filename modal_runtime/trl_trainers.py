"""TRL trainer integration for RL algorithms."""
from typing import Dict, Any, Optional, List, Callable
from datasets import Dataset


def create_dpo_trainer(
    model,
    ref_model,
    tokenizer,
    train_dataset: Dataset,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    max_length: int = 512,
    max_prompt_length: int = 256,
    num_train_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 10,
):
    """Create DPO trainer with TRL."""
    from trl import DPOTrainer, DPOConfig
    
    config = DPOConfig(
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        remove_unused_columns=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # TRL will create from model if None
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    
    return trainer


def create_ppo_trainer(
    model,
    tokenizer,
    learning_rate: float = 1.41e-5,
    batch_size: int = 16,
    mini_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    ppo_epochs: int = 4,
    max_length: int = 512,
):
    """Create PPO trainer with TRL."""
    
    from trl import PPOTrainer, PPOConfig
    
    config = PPOConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ppo_epochs=ppo_epochs,
        max_length=max_length,
    )
    
    trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    return trainer


def create_grpo_trainer(
    model,
    tokenizer,
    reward_funcs: List[Callable],
    train_dataset: Dataset,
    learning_rate: float = 1e-6,
    beta: float = 0.0,
    num_generations: int = 8,
    max_prompt_length: int = 1024,
    max_completion_length: int = 2048,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    loss_type: str = "grpo",
    scale_rewards: Optional[str] = None,
    epsilon: float = 0.2,
    epsilon_high: Optional[float] = None,
    mask_truncated_completions: bool = False,
    use_vllm: bool = False,
    vllm_importance_sampling_correction: bool = True,
    vllm_importance_sampling_cap: float = 2.0,
):
    """Create GRPO trainer with TRL."""
    from trl import GRPOTrainer, GRPOConfig
    
    config = GRPOConfig(
        learning_rate=learning_rate,
        beta=beta,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        epsilon=epsilon,
        # Lite PPO (batch-scaled rewards)
        scale_rewards=scale_rewards,
        # DAPO-specific
        epsilon_high=epsilon_high,
        mask_truncated_completions=mask_truncated_completions,
        # vLLM + TIS
        use_vllm=use_vllm,
        vllm_importance_sampling_correction=vllm_importance_sampling_correction,
        vllm_importance_sampling_cap=vllm_importance_sampling_cap,
    )
    
    trainer = GRPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
    )
    
    return trainer


def get_soft_overlong_punishment(
    max_completion_len: int = 20480,
    soft_punish_cache: int = 4096,
) -> Callable:
    """Create soft overlong punishment reward function for DAPO."""
    from trl import get_soft_overlong_punishment as trl_sop
    return trl_sop(
        max_completion_len=max_completion_len,
        soft_punish_cache=soft_punish_cache,
    )


def create_reward_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 10,
):
    """Create Reward Model trainer with TRL."""
    from trl import RewardTrainer, RewardConfig
    
    config = RewardConfig(
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
    )
    
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
    )
    
    return trainer


def prepare_dpo_dataset(
    preference_pairs: List[Dict[str, str]],
) -> Dataset:
    """Convert preference pairs to TRL DPO format."""
    prompts = [p["prompt"] for p in preference_pairs]
    chosen = [p["chosen"] for p in preference_pairs]
    rejected = [p["rejected"] for p in preference_pairs]
    
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
    })
    
    return dataset


def prepare_reward_dataset(
    preference_pairs: List[Dict[str, str]],
) -> Dataset:
    """Convert preference pairs to TRL Reward Model format."""
    chosen = [p["chosen"] for p in preference_pairs]
    rejected = [p["rejected"] for p in preference_pairs]
    
    dataset = Dataset.from_dict({
        "chosen": chosen,
        "rejected": rejected,
    })
    
    return dataset


# Helper function for chat template formatting
def format_chat_for_dpo(
    preference_pairs: List[Dict[str, Any]],
    tokenizer,
) -> List[Dict[str, str]]:
    """Format chat-style preference pairs for DPO."""
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError("Tokenizer does not support chat templates")
    
    formatted_pairs = []
    
    for pair in preference_pairs:
        # Extract messages
        messages_chosen = pair["messages_chosen"]
        messages_rejected = pair["messages_rejected"]
        
        # Separate prompt (user messages) from responses
        # Assume last message is assistant response
        prompt_messages = messages_chosen[:-1]
        
        # Format prompt
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Format chosen and rejected (full conversations)
        chosen = tokenizer.apply_chat_template(
            messages_chosen,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        rejected = tokenizer.apply_chat_template(
            messages_rejected,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        formatted_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    
    return formatted_pairs
