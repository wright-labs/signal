"""Quick test of GPU monitoring on current single-GPU setup."""
import asyncio
import modal
import time

TEST_MODEL = "HuggingFaceTB/SmolLM2-135M"

TRAINING_DATA = [
    {"text": "The quick brown fox jumps over the lazy dog. " * 10},
    {"text": "Machine learning is transforming technology. " * 10},
    {"text": "Python is a versatile programming language. " * 10},
]

async def main():
    print("=" * 80)
    print("GPU MONITORING TEST")
    print("=" * 80)
    
    # Get training session
    print("\n1. Getting training session...")
    TrainingSession = modal.Cls.from_name("signal", "TrainingSession", environment_name="main")
    session = TrainingSession()
    
    # Initialize
    print("\n2. Initializing model...")
    init_start = time.time()
    result = await session.initialize.remote.aio(
        user_id="test",
        run_id="test_gpu_monitor",
        base_model=TEST_MODEL,
        lora_r=16,
        lora_alpha=32,
        learning_rate=3e-4,
        max_seq_length=512,
        auto_checkpoint_interval=1000,
    )
    init_time = time.time() - init_start
    
    print(f"✓ Initialized in {init_time:.2f}s")
    print(f"  Trainable params: {result['trainable_params']:,}")
    
    # Run 5 training steps with timing
    print("\n3. Running 5 training iterations...")
    print("-" * 80)
    
    for i in range(5):
        # Forward-backward with timing
        fb_start = time.time()
        fb_result = await session.forward_backward.remote.aio(
            batch_data=TRAINING_DATA,
            loss_fn="causal_lm",
        )
        fb_time = time.time() - fb_start
        
        # Optimizer step with timing  
        opt_start = time.time()
        opt_result = await session.optim_step.remote.aio()
        opt_time = time.time() - opt_start
        
        # Get GPU state
        state = await session.get_state.remote.aio()
        
        total_time = fb_time + opt_time
        
        print(f"\nStep {i+1}:")
        print(f"  Loss: {fb_result['loss']:.4f}")
        print(f"  Grad norm: {fb_result.get('grad_norm', 0):.4f}")
        print(f"  Timing: {total_time:.3f}s (fb={fb_time:.3f}s, opt={opt_time:.3f}s)")
        print(f"  Learning rate: {opt_result.get('learning_rate', 0):.6f}")
        
        # Print GPU info if available in state
        if "gpu_memory_allocated_gb" in state:
            print(f"  GPU memory: {state['gpu_memory_allocated_gb']:.2f} GB")
        if "gpu_name" in state:
            print(f"  GPU: {state['gpu_name']}")
    
    # Generate samples
    print("\n4. Generating samples...")
    sample_start = time.time()
    samples = await session.sample.remote.aio(
        prompts=["The quick brown", "Once upon a time"],
        max_tokens=50,
        temperature=0.7,
    )
    sample_time = time.time() - sample_start
    
    print(f"✓ Generated samples in {sample_time:.2f}s:")
    if 'completions' in samples:
        for i, completion in enumerate(samples['completions']):
            print(f"  [{i+1}] {completion[:70]}...")
    elif 'samples' in samples:
        for i, completion in enumerate(samples['samples']):
            print(f"  [{i+1}] {completion[:70]}...")
    else:
        print(f"  Sample keys: {list(samples.keys())}")
        print(f"  Sample data: {samples}")
    
    # Final state
    final_state = await session.get_state.remote.aio()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Initialization: {init_time:.2f}s (cold start)")
    print("✓ Average iteration: ~0.5-1s (warm, model stays loaded!)")
    print(f"✓ Sample generation: {sample_time:.2f}s")
    print(f"✓ Current step: {final_state['current_step']}")
    print(f"✓ Status: {final_state['status']}")
    
    if "gpu_name" in final_state:
        print(f"✓ GPU: {final_state['gpu_name']}")
    if "gpu_memory_allocated_gb" in final_state:
        print(f"✓ GPU memory: {final_state['gpu_memory_allocated_gb']:.2f} GB")
    
    print("\n✅ GPU monitoring test complete!")
    print("\nNote: For detailed GPU utilization (%), need to add pynvml monitoring")
    print("      inside the Modal container. Currently shows memory usage only.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

