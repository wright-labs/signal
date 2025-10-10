"""Basic asynchronous example of using the Signal SDK."""

import asyncio
from frontier_signal import AsyncSignalClient

async def main():
    # Use async context manager for automatic cleanup
    async with AsyncSignalClient(
        api_key="sk-your-api-key-here",  # Replace with your API key
        base_url="http://localhost:8000",  # Use production URL: https://api.frontier-signal.com
    ) as client:
        # List available models
        print("Fetching available models...")
        models = await client.list_models()
        print(f"Available models: {models}\n")
        
        # Create a training run
        print("Creating training run...")
        run = await client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            lora_r=32,
            lora_alpha=64,
            learning_rate=3e-4,
            max_seq_length=2048,
        )
        print(f"Created run: {run.run_id}\n")
        
        # Prepare training data with chat format
        batch = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain Python in simple terms."},
                    {"role": "assistant", "content": "Python is a popular programming language..."}
                ]
            },
        ]
        
        # Training loop
        print("Starting training...")
        num_steps = 10
        
        for step in range(num_steps):
            # Forward-backward pass
            result = await run.forward_backward(batch=batch)
            loss = result['loss']
            grad_norm = result.get('grad_norm', 'N/A')
            
            print(f"Step {step + 1}/{num_steps}: Loss = {loss:.4f}, Grad Norm = {grad_norm}")
            
            # Optimizer step
            optim_result = await run.optim_step()
            
            # Sample from model every 3 steps
            if (step + 1) % 3 == 0:
                print("\nGenerating sample...")
                samples = await run.sample(
                    prompts=["Explain quantum computing in one sentence:"],
                    max_tokens=50,
                    temperature=0.7,
                )
                print(f"Sample output: {samples['outputs'][0]}\n")
        
        # Get final status
        print("\nFetching run status...")
        status = await run.get_status()
        print(f"Run status: {status['status']}")
        print(f"Current step: {status['current_step']}")
        
        # Get metrics
        print("\nFetching metrics...")
        metrics = await run.get_metrics()
        print(f"Metrics count: {len(metrics.get('metrics', []))}")
        
        # Save final model
        print("\nSaving model state...")
        artifact = await run.save_state(
            mode="adapter",
            push_to_hub=False,
        )
        print(f"Model saved to: {artifact['checkpoint_path']}")
        print(f"Artifact URI: {artifact['artifact_uri']}")
        
        print("\nTraining complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
