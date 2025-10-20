"""
Example: Training with Direct Preference Optimization (DPO)

DPO is a reinforcement learning technique that directly optimizes a language model
from preference data (chosen vs rejected responses) without needing a reward model.

This example shows how to:
1. Format preference data for DPO training
2. Create a training run
3. Train with DPO loss
4. Monitor training metrics
"""

import rewardsignal as rs
from typing import List, Dict


def create_dpo_dataset() -> List[Dict[str, str]]:
    """
    Create a sample DPO dataset with preference pairs.
    
    Each example has:
    - prompt: The input question or instruction
    - chosen: The preferred/better response
    - rejected: The worse/rejected response
    """
    return [
        {
            "prompt": "How do I learn Python programming?",
            "chosen": "Start with the basics: learn syntax, data types, and control flow. "
                     "Then practice with small projects like a calculator or to-do list. "
                     "Use resources like Python.org documentation, online courses, and coding challenges.",
            "rejected": "Just memorize all the syntax and you'll be fine.",
        },
        {
            "prompt": "What's the best way to debug code?",
            "chosen": "Use a debugger to step through your code line by line. Set breakpoints "
                     "at suspicious locations, inspect variable values, and trace the execution flow. "
                     "Also write unit tests to catch bugs early.",
            "rejected": "Add print() statements everywhere and hope you find the bug.",
        },
        {
            "prompt": "Should I use tabs or spaces for indentation?",
            "chosen": "The Python community standard (PEP 8) recommends using 4 spaces per "
                     "indentation level. Most modern editors can convert tabs to spaces automatically.",
            "rejected": "It doesn't matter at all, just use whatever you want.",
        },
        {
            "prompt": "What are some good practices for writing functions?",
            "chosen": "Functions should be small and focused on a single task. Use descriptive names, "
                     "add docstrings to explain what the function does, and keep the number of parameters "
                     "reasonable (usually under 5). Follow the Single Responsibility Principle.",
            "rejected": "Make functions as long as possible to save space. Who needs documentation anyway?",
        },
        {
            "prompt": "How do I handle errors in Python?",
            "chosen": "Use try-except blocks to catch specific exceptions. Handle errors gracefully "
                     "with informative messages, and only catch exceptions you can actually handle. "
                     "Consider using finally blocks for cleanup code.",
            "rejected": "Just wrap everything in try-except and pass on all errors. What could go wrong?",
        },
    ]


def main():
    """Main DPO training example."""
    
    # Initialize client
    client = rs.Client(
        api_key="your-api-key-here",
        base_url="https://api.rewardsignal.com/v1"
    )
    
    # Create a training run
    print("Creating training run...")
    run = client.training.create(
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=32,
        lora_alpha=64,
        learning_rate=5e-5,  # Lower learning rate for DPO
        max_seq_length=2048,
    )
    
    print(f"✓ Run created: {run.run_id}")
    print(f"  Model: {run.base_model}")
    print(f"  Status: {run.status}")
    
    # Prepare DPO dataset
    dpo_data = create_dpo_dataset()
    print(f"\n📊 Training with {len(dpo_data)} preference pairs")
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
        
        # Forward-backward pass with DPO loss
        result = client.training.forward_backward(
            run_id=run.run_id,
            batch_data=dpo_data,
            loss_fn="dpo",  # Use DPO loss function
            loss_kwargs={
                "beta": 0.1,  # KL penalty coefficient (higher = stay closer to base model)
                "label_smoothing": 0.0,  # Conservative DPO (0.0 = standard DPO)
            }
        )
        
        print(f"  Loss: {result.loss:.4f}")
        
        # Print DPO-specific metrics
        if "preference_accuracy" in result.grad_stats:
            print(f"  Preference Accuracy: {result.grad_stats['preference_accuracy']:.2%}")
        if "reward_margin" in result.grad_stats:
            print(f"  Reward Margin: {result.grad_stats['reward_margin']:.4f}")
        if "implicit_reward" in result.grad_stats:
            print(f"  Implicit Reward: {result.grad_stats['implicit_reward']:.4f}")
        
        # Optimizer step
        optim_result = client.training.optim_step(run_id=run.run_id)
        print(f"  Step: {optim_result.step}, LR: {optim_result.learning_rate:.2e}")
        
        # Sample from the model to see improvements
        if (epoch + 1) % 1 == 0:
            print("\n  📝 Sample generation:")
            test_prompt = "What's the best way to learn programming?"
            
            samples = client.training.sample(
                run_id=run.run_id,
                prompts=[test_prompt],
                max_tokens=100,
                temperature=0.7,
            )
            
            print(f"  Prompt: {test_prompt}")
            print(f"  Response: {samples.outputs[0]}")
    
    # Save the trained model
    print("\n💾 Saving model...")
    save_result = client.training.save_state(
        run_id=run.run_id,
        mode="adapter",  # Save LoRA adapter
    )
    
    print(f"✓ Model saved to: {save_result.artifact_uri}")
    if save_result.s3_uri:
        print(f"  S3 URI: {save_result.s3_uri}")
    
    # Get final run status
    status = client.training.get_status(run_id=run.run_id)
    print("\n✅ Training complete!")
    print(f"  Total steps: {status.current_step}")
    print(f"  Cost: ${status.cost_so_far:.4f}")
    
    return run.run_id


def advanced_dpo_example():
    """
    Advanced DPO training with:
    - Batch processing of larger datasets
    - Gradient accumulation
    - Custom beta scheduling
    - Periodic checkpointing
    """
    client = rs.Client(api_key="your-api-key-here")
    
    # Create run
    run = client.training.create(
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        lora_r=64,  # Higher rank for more capacity
        learning_rate=1e-5,  # Very low LR for DPO
    )
    
    # Load larger dataset (you would load from file/database)
    full_dataset = create_dpo_dataset() * 100  # Simulate larger dataset
    
    batch_size = 8
    accumulation_steps = 4  # Effective batch size = 8 * 4 = 32
    
    # Beta scheduling: start high, decay over time
    initial_beta = 0.5
    final_beta = 0.1
    num_steps = len(full_dataset) // batch_size
    
    print(f"Training with {len(full_dataset)} examples")
    print(f"Batch size: {batch_size}, Accumulation: {accumulation_steps}")
    
    step = 0
    for batch_start in range(0, len(full_dataset), batch_size):
        batch = full_dataset[batch_start:batch_start + batch_size]
        
        # Calculate current beta (linear decay)
        progress = step / num_steps
        current_beta = initial_beta + (final_beta - initial_beta) * progress
        
        # Forward-backward with gradient accumulation
        accumulate = (step % accumulation_steps) != 0
        
        result = client.training.forward_backward(
            run_id=run.run_id,
            batch_data=batch,
            loss_fn="dpo",
            loss_kwargs={"beta": current_beta},
            accumulate=accumulate,
        )
        
        # Optimizer step after accumulating enough gradients
        if not accumulate:
            client.training.optim_step(run_id=run.run_id)
            print(f"Step {step // accumulation_steps}: Loss={result.loss:.4f}, Beta={current_beta:.3f}")
        
        # Checkpoint every 100 steps
        if step % 100 == 0 and step > 0:
            client.training.save_state(run_id=run.run_id, mode="adapter")
            print(f"  💾 Checkpoint saved at step {step}")
        
        step += 1
    
    print("✅ Advanced DPO training complete!")


if __name__ == "__main__":
    # Run basic example
    run_id = main()
    
    print("\n" + "="*60)
    print("For advanced usage, see the advanced_dpo_example() function")
    print("="*60)

