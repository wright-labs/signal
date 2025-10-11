"""Example: Advanced training with TrainingClient.

This example demonstrates how to use the advanced TrainingClient
for fine-grained control over training with state tracking.
"""

from frontier_signal import SignalClient


def main():
    # Initialize client
    client = SignalClient(api_key="your-api-key-here")
    
    # Create a training run
    print("Creating training run...")
    run = client.create_run(
        base_model="Qwen/Qwen2.5-3B",
        lora_r=8,
        lora_alpha=16,
        learning_rate=5e-4,
        max_seq_length=512,
    )
    print(f"✓ Created run: {run.run_id}")
    
    # Get specialized training client with custom config
    print("\nInitializing training client...")
    training = client.training(
        run_id=run.run_id,
        timeout=7200,  # 2 hours for long training
        max_retries=3,
    )
    
    # Example 1: Fine-grained control with conditional optimizer steps
    print("\n=== Example 1: Fine-grained control ===")
    
    # Simulate a dataloader
    dataloader = [
        [{"text": "The capital of France is Paris."}],
        [{"text": "Python is a programming language."}],
        [{"text": "Machine learning is a subset of AI."}],
    ]
    
    for batch in dataloader:
        # Compute gradients
        result = training.forward_backward(batch)
        print(f"Loss: {result['loss']:.4f}, Grad norm: {result['grad_norm']:.4f}")
        
        # Conditional optimizer step (e.g., gradient clipping)
        if result['grad_norm'] < 10.0:
            training.optim_step()
            print(f"✓ Optimizer step applied")
        else:
            print(f"✗ Skipping step: gradient norm too high")
    
    # Example 2: Using convenience method train_batch
    print("\n=== Example 2: Convenience method train_batch ===")
    
    for i, batch in enumerate(dataloader):
        result = training.train_batch(
            batch_data=batch,
            learning_rate=1e-4,  # Override learning rate
        )
        print(f"Step {result['step']}: loss={result['loss']:.4f}")
    
    # Example 3: Training full epoch with progress tracking
    print("\n=== Example 3: Training epoch with progress ===")
    
    result = training.train_epoch(
        dataloader=dataloader,
        progress=True,  # Shows tqdm progress bar
    )
    
    print(f"\nEpoch complete:")
    print(f"  Batches processed: {result['num_batches']}")
    print(f"  Average loss: {result['avg_loss']:.4f}")
    print(f"  Average grad norm: {result['avg_grad_norm']:.4f}")
    print(f"  Final step: {result['final_step']}")
    
    # Example 4: Accessing training metrics
    print("\n=== Example 4: Training metrics ===")
    
    metrics = training.get_metrics()
    print(f"Current step: {metrics['current_step']}")
    print(f"Total losses recorded: {len(metrics['loss_history'])}")
    print(f"Average loss: {metrics['avg_loss']:.4f}")
    print(f"Average grad norm: {metrics['avg_grad_norm']:.4f}")
    
    # Save checkpoint
    print("\n=== Saving checkpoint ===")
    result = training.save_checkpoint(
        mode="adapter",
        push_to_hub=False,
    )
    print(f"✓ Checkpoint saved: {result.get('artifact_url', 'saved locally')}")
    
    # Example 5: Using context manager for automatic cleanup
    print("\n=== Example 5: Context manager ===")
    
    with client.training(run.run_id) as training:
        print("Training in context manager...")
        for batch in dataloader[:2]:  # Train on 2 batches
            training.train_batch(batch)
        print("✓ Training complete, session closed automatically")
    
    print("\n✅ Advanced training examples complete!")


if __name__ == "__main__":
    main()

