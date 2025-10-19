"""
Example: Supervised Fine-Tuning with Signal API

This example demonstrates how to fine-tune a language model using
the Signal API with the four training primitives.
"""

from client.rewardsignal import SignalClient


def main():
    # =============================================================================
    # 1. Initialize Client
    # =============================================================================
    
    print("Initializing Signal client...")
    client = SignalClient(
        api_key="sk-...",  # Replace with your API key
        base_url="http://localhost:8000"
    )
    
    # List available models
    print("\nAvailable models:")
    models = client.list_models()
    for model in models[:5]:  # Show first 5
        print(f"  - {model}")
    
    # =============================================================================
    # 2. Create Training Run
    # =============================================================================
    
    print("\nCreating training run...")
    run = client.create_run(
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        optimizer="adamw_8bit",
        learning_rate=3e-4,
        weight_decay=0.01,
        max_seq_length=2048,
        bf16=True,
        gradient_checkpointing=True,
    )
    
    print(f"✓ Created run: {run.run_id}")
    
    # =============================================================================
    # 3. Prepare Training Data
    # =============================================================================
    
    # Example 1: Simple text completion
    text_batch = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning is the study of algorithms that improve through experience."},
        {"text": "Python is a high-level, interpreted programming language."},
    ]
    
    # Example 2: Chat format (for instruction tuning)
    chat_batch = [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain quantum computing in simple terms."},
                {"role": "assistant", "content": "Quantum computing uses quantum mechanics to process information differently than classical computers, potentially solving certain problems much faster."}
            ]
        },
    ]
    
    # Choose which format to use
    training_batch = text_batch  # or chat_batch
    
    # =============================================================================
    # 4. Training Loop
    # =============================================================================
    
    print("\nStarting training loop...")
    num_steps = 20
    
    for step in range(num_steps):
        # Forward-backward pass: compute gradients
        fb_result = run.forward_backward(batch=training_batch)
        
        loss = fb_result["loss"]
        grad_norm = fb_result.get("grad_norm", 0.0)
        
        print(f"Step {step:3d} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
        
        # Optimizer step: update weights
        optim_result = run.optim_step()
        
        # Sample from model every 5 steps
        if step % 5 == 0:
            print(f"\n{'='*60}")
            print(f"Generating sample at step {step}...")
            print(f"{'='*60}")
            
            sample_result = run.sample(
                prompts=["The meaning of life is"],
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )
            
            print(f"Output: {sample_result['outputs'][0]}")
            print(f"{'='*60}\n")
    
    # =============================================================================
    # 5. Save Final Model
    # =============================================================================
    
    print("\nSaving model...")
    
    # Save LoRA adapter only
    adapter_result = run.save_state(mode="adapter")
    print(f"✓ Saved LoRA adapter to: {adapter_result['checkpoint_path']}")
    
    # Optionally save merged model
    # merged_result = run.save_state(mode="merged")
    # print(f"✓ Saved merged model to: {merged_result['checkpoint_path']}")
    
    # Optionally push to HuggingFace Hub
    # hub_result = run.save_state(
    #     mode="adapter",
    #     push_to_hub=True,
    #     hub_model_id="your-username/your-model-name"
    # )
    
    # =============================================================================
    # 6. Check Final Status
    # =============================================================================
    
    print("\nFinal run status:")
    status = run.get_status()
    print(f"  Run ID: {status['run_id']}")
    print(f"  Status: {status['status']}")
    print(f"  Current Step: {status['current_step']}")
    print(f"  Model: {status['base_model']}")
    
    print("\nTraining metrics:")
    metrics = run.get_metrics()
    if metrics['metrics']:
        last_5_metrics = metrics['metrics'][-5:]
        for m in last_5_metrics:
            print(f"  Step {m.get('step', 'N/A'):3d} | Loss: {m.get('loss', 0.0):.4f}")
    
    print("\n✓ Training complete!")


def example_with_gradient_accumulation():
    """Example showing gradient accumulation."""
    
    client = SignalClient(
        api_key="sk-...",
        base_url="http://localhost:8000"
    )
    
    run = client.create_run(
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=32,
        learning_rate=3e-4,
    )
    
    # Prepare multiple small batches
    batch_1 = [{"text": "Example 1"}]
    batch_2 = [{"text": "Example 2"}]
    batch_3 = [{"text": "Example 3"}]
    
    # Accumulate gradients across batches
    print("Accumulating gradients...")
    
    # First batch: don't accumulate (clear previous gradients)
    result_1 = run.forward_backward(batch=batch_1, accumulate=False)
    print(f"Batch 1 loss: {result_1['loss']:.4f}")
    
    # Subsequent batches: accumulate
    result_2 = run.forward_backward(batch=batch_2, accumulate=True)
    print(f"Batch 2 loss: {result_2['loss']:.4f}")
    
    result_3 = run.forward_backward(batch=batch_3, accumulate=True)
    print(f"Batch 3 loss: {result_3['loss']:.4f}")
    
    # Apply accumulated gradients
    run.optim_step()
    print("✓ Applied accumulated gradients")


def example_with_learning_rate_schedule():
    """Example with custom learning rate schedule."""
    
    client = SignalClient(
        api_key="sk-...",
        base_url="http://localhost:8000"
    )
    
    run = client.create_run(
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=32,
        learning_rate=3e-4,  # Initial LR
    )
    
    batch = [{"text": "Training example"}]
    
    # Custom learning rate schedule
    num_steps = 20
    warmup_steps = 5
    
    for step in range(num_steps):
        # Forward-backward
        result = run.forward_backward(batch=batch)
        
        # Compute learning rate with warmup
        if step < warmup_steps:
            # Linear warmup
            lr = 3e-4 * (step + 1) / warmup_steps
        else:
            # Cosine decay
            import math
            progress = (step - warmup_steps) / (num_steps - warmup_steps)
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply optimizer step with custom LR
        run.optim_step(learning_rate=lr)
        
        print(f"Step {step:3d} | Loss: {result['loss']:.4f} | LR: {lr:.6f}")


if __name__ == "__main__":
    print("=" * 80)
    print("Signal API - Supervised Fine-Tuning Example")
    print("=" * 80)
    
    # Run main example
    main()
    
    # Uncomment to run other examples:
    # example_with_gradient_accumulation()
    # example_with_learning_rate_schedule()

