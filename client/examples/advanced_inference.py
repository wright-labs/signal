"""Example: Advanced inference with InferenceClient.

This example demonstrates how to use the advanced InferenceClient
for batched generation, caching, and checkpoint comparison.
"""

from frontier_signal import SignalClient


def main():
    # Initialize client
    client = SignalClient(api_key="your-api-key-here")
    
    # Assume we have a trained run
    run_id = "your-run-id-here"
    
    # Example 1: Basic inference with specific checkpoint
    print("=== Example 1: Basic inference ===\n")
    
    inference = client.inference(
        run_id=run_id,
        step=100,  # Use checkpoint at step 100
        timeout=30,
    )
    
    outputs = inference.sample(
        prompts=["What is machine learning?"],
        max_tokens=100,
        temperature=0.7,
    )
    
    print(f"Prompt: What is machine learning?")
    print(f"Output: {outputs[0]}\n")
    
    # Example 2: Batched inference
    print("=== Example 2: Batched inference ===\n")
    
    inference = client.inference(
        run_id=run_id,
        step=100,
        batch_size=32,  # Process 32 prompts at a time
    )
    
    # Many prompts
    prompts = [
        "Explain quantum computing",
        "What is artificial intelligence?",
        "How does a neural network work?",
        "What is deep learning?",
        "Explain natural language processing",
    ] * 10  # 50 prompts total
    
    print(f"Generating outputs for {len(prompts)} prompts...")
    outputs = inference.batch_sample(
        prompts=prompts,
        max_tokens=50,
        temperature=0.7,
    )
    
    print(f"✓ Generated {len(outputs)} outputs")
    print(f"First output: {outputs[0][:100]}...\n")
    
    # Example 3: Caching for repeated prompts
    print("=== Example 3: Response caching ===\n")
    
    inference = client.inference(run_id=run_id, step=100)
    
    # Enable caching
    inference.enable_cache()
    
    prompt = "What is the capital of France?"
    
    # First call hits the API
    print("First call (hits API)...")
    import time
    start = time.time()
    output1 = inference.sample([prompt], max_tokens=50)
    elapsed1 = time.time() - start
    print(f"✓ Time: {elapsed1:.3f}s")
    
    # Second call returns cached result (instant)
    print("\nSecond call (cached)...")
    start = time.time()
    output2 = inference.sample([prompt], max_tokens=50)
    elapsed2 = time.time() - start
    print(f"✓ Time: {elapsed2:.3f}s (cached)")
    
    assert output1 == output2
    
    # Check cache stats
    stats = inference.get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Enabled: {stats['cache_enabled']}")
    print(f"  Size: {stats['cache_size']}")
    
    # Clear cache
    inference.clear_cache()
    print("✓ Cache cleared\n")
    
    # Example 4: Comparing different checkpoints
    print("=== Example 4: Checkpoint comparison ===\n")
    
    # Early checkpoint
    inference_early = client.inference(run_id=run_id, step=10)
    
    # Late checkpoint
    inference_late = client.inference(run_id=run_id, step=1000)
    
    prompt = "Explain machine learning"
    
    output_early = inference_early.sample([prompt], max_tokens=100)
    output_late = inference_late.sample([prompt], max_tokens=100)
    
    print(f"Prompt: {prompt}\n")
    print(f"Early checkpoint (step 10):")
    print(f"{output_early[0]}\n")
    print(f"Late checkpoint (step 1000):")
    print(f"{output_late[0]}\n")
    
    # Example 5: Using context manager
    print("=== Example 5: Context manager ===\n")
    
    with client.inference(run_id=run_id, step=100) as inference:
        print("Generating in context manager...")
        outputs = inference.sample(
            prompts=["Hello world"],
            max_tokens=50,
        )
        print(f"Output: {outputs[0]}")
        print("✓ Session closed automatically\n")
    
    # Example 6: Temperature sweeping
    print("=== Example 6: Temperature sweep ===\n")
    
    inference = client.inference(run_id=run_id, step=100)
    prompt = "Write a creative story about"
    
    temperatures = [0.1, 0.5, 0.9]
    
    for temp in temperatures:
        output = inference.sample(
            prompts=[prompt],
            max_tokens=50,
            temperature=temp,
        )
        print(f"Temperature {temp}:")
        print(f"{output[0]}\n")
    
    print("✅ Advanced inference examples complete!")


if __name__ == "__main__":
    main()

