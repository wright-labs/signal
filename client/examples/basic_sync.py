"""Basic synchronous example of using the Frontier Signal SDK."""

from frontier_signal import SignalClient

# Initialize client
client = SignalClient(
    api_key="sk-your-api-key-here",  # Replace with your API key
    base_url="http://localhost:8000",  # Use production URL: https://api.frontier-signal.com
)

def main():
    # List available models
    print("Fetching available models...")
    models = client.list_models()
    print(f"Available models: {models}\n")
    
    # Create a training run
    print("Creating training run...")
    run = client.create_run(
        base_model="meta-llama/Llama-3.2-3B",
        lora_r=32,
        lora_alpha=64,
        learning_rate=3e-4,
        max_seq_length=2048,
    )
    print(f"Created run: {run.run_id}\n")
    
    # Prepare training data
    batch = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning is transforming technology."},
        {"text": "Python is a versatile programming language."},
        {"text": "The future of AI is bright and promising."},
    ]
    
    # Training loop
    print("Starting training...")
    num_steps = 10
    
    for step in range(num_steps):
        # Forward-backward pass
        result = run.forward_backward(batch=batch)
        loss = result['loss']
        grad_norm = result.get('grad_norm', 'N/A')
        
        print(f"Step {step + 1}/{num_steps}: Loss = {loss:.4f}, Grad Norm = {grad_norm}")
        
        # Optimizer step
        optim_result = run.optim_step()
        
        # Sample from model every 3 steps
        if (step + 1) % 3 == 0:
            print("\nGenerating sample...")
            samples = run.sample(
                prompts=["The meaning of life is"],
                max_tokens=50,
                temperature=0.7,
            )
            print(f"Sample output: {samples['outputs'][0]}\n")
    
    # Get final status
    print("\nFetching run status...")
    status = run.get_status()
    print(f"Run status: {status['status']}")
    print(f"Current step: {status['current_step']}")
    
    # Save final model
    print("\nSaving model state...")
    artifact = run.save_state(
        mode="adapter",
        push_to_hub=False,
    )
    print(f"Model saved to: {artifact['checkpoint_path']}")
    print(f"Artifact URI: {artifact['artifact_uri']}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise
