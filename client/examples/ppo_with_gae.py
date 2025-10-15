"""Example: PPO training with Generalized Advantage Estimation (GAE).

This example demonstrates how to use Signal's enhanced PPO implementation
with GAE for variance reduction and comprehensive metrics tracking.
"""

import asyncio
from rewardsignal import AsyncSignalClient


async def main():
    # Initialize client
    client = AsyncSignalClient(api_key="your-api-key-here")
    
    # Create training run with PPO-optimized settings
    run = await client.create_run(
        base_model="meta-llama/Llama-3.2-1B",
        lora_r=32,
        lora_alpha=64,
        learning_rate=1e-5,  # Lower LR for PPO stability
        optimizer="adamw_8bit",
    )
    
    run_id = run["run_id"]
    print(f"Created run: {run_id}")
    
    # Get training client
    training_client = client.get_training_client(run_id)
    
    # Example PPO training batch with GAE
    # In practice, you'd collect these from policy rollouts
    ppo_batch = [
        {
            "text": "User: What is machine learning?\nAssistant: Machine learning is...",
            "rewards": [0.5, 0.6, 0.7, 0.8, 0.9],  # Per-token rewards
            "values": [0.4, 0.5, 0.6, 0.7, 0.8],  # Value function estimates
            "old_log_probs": [-1.2, -1.3, -1.1, -1.4, -1.2],  # From policy rollout
        }
    ]
    
    # Train with enhanced PPO + GAE
    result = await training_client.forward_backward(
        batch_data=ppo_batch,
        loss_fn="enhanced_ppo",
        loss_kwargs={
            "use_gae": True,  # Enable GAE
            "gamma": 0.99,  # Discount factor
            "gae_lambda": 0.95,  # GAE lambda (higher = more variance, less bias)
            "clip_epsilon": 0.2,  # PPO clip parameter
            "value_loss_coef": 0.5,  # Value function loss weight
            "entropy_coef": 0.01,  # Entropy bonus for exploration
            "beta": 0.01,  # KL penalty coefficient
        }
    )
    
    # Check comprehensive metrics
    print("\n=== PPO Training Metrics ===")
    print(f"Loss: {result['loss']:.4f}")
    
    if "metrics" in result:
        metrics = result["metrics"]
        print(f"Policy Loss: {metrics.get('policy_loss', 0):.4f}")
        print(f"Value Loss: {metrics.get('value_loss', 0):.4f}")
        print(f"Entropy: {metrics.get('entropy', 0):.4f}")
        print(f"KL Divergence: {metrics.get('kl_divergence', 0):.6f}")
        print(f"Clip Fraction: {metrics.get('clip_fraction', 0):.3f}")
        print(f"Advantage Mean: {metrics.get('advantage_mean', 0):.4f}")
        print(f"Advantage Std: {metrics.get('advantage_std', 0):.4f}")
        print(f"Explained Variance: {metrics.get('explained_variance', 0):.3f}")
    
    # Apply optimizer update
    opt_result = await training_client.optim_step(learning_rate=1e-5)
    print(f"\nOptimizer step complete: step {opt_result['step']}")
    
    # Full training loop
    print("\n=== Full PPO Training Loop ===")
    for epoch in range(10):
        # In practice: collect rollouts, compute rewards, estimate values
        # For demo: using same batch
        
        fb_result = await training_client.forward_backward(
            batch_data=ppo_batch,
            loss_fn="enhanced_ppo",
            loss_kwargs={
                "use_gae": True,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "value_loss_coef": 0.5,
                "entropy_coef": 0.01,
            }
        )
        
        opt_result = await training_client.optim_step()
        
        print(f"Epoch {epoch+1}: Loss={fb_result['loss']:.4f}, "
              f"KL={fb_result.get('metrics', {}).get('kl_divergence', 0):.6f}, "
              f"Step={opt_result['step']}")
    
    # Save checkpoint
    checkpoint = await training_client.save_checkpoint(mode="adapter")
    print(f"\nCheckpoint saved: {checkpoint['artifact_uri']}")
    
    print("\n✓ PPO training with GAE complete!")


if __name__ == "__main__":
    asyncio.run(main())

