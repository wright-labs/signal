"""Test multi-GPU training with GPU utilization monitoring.

This script tests:
1. Training on multiple GPUs (DataParallel)
2. GPU utilization monitoring per GPU
3. Memory usage per GPU
4. Performance comparison (1 GPU vs multi-GPU)
"""
import asyncio
import time
import modal
import statistics

# Configuration
TEST_MODEL = "HuggingFaceTB/SmolLM2-135M"
TEST_PROMPTS = [
    "The quick brown fox",
    "Once upon a time",
    "In a galaxy far away",
    "def fibonacci(n):",
]

# Test data
TRAINING_DATA = [
    {"text": "The quick brown fox jumps over the lazy dog. " * 10},
    {"text": "Once upon a time, in a land far away, there lived a brave knight. " * 10},
    {"text": "Python is a high-level programming language known for its simplicity. " * 10},
    {"text": "Machine learning is a subset of artificial intelligence. " * 10},
]


async def test_single_gpu():
    """Test training on single GPU with monitoring."""
    print("\n" + "=" * 80)
    print("TEST 1: SINGLE GPU TRAINING")
    print("=" * 80)
    
    # Get training session class
    TrainingSession = modal.Cls.from_name("signal", "TrainingSession", environment_name="main")
    session = TrainingSession()
    
    # Initialize with single GPU
    print("\n1. Initializing on single L40S GPU...")
    start_time = time.time()
    
    result = await session.initialize.remote.aio(
        user_id="test",
        run_id="test_single_gpu",
        base_model=TEST_MODEL,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        learning_rate=3e-4,
        max_seq_length=512,
        gradient_checkpointing=True,
        auto_checkpoint_interval=1000,
    )
    
    init_time = time.time() - start_time
    print(f"✓ Initialized in {init_time:.2f}s")
    print(f"   Trainable params: {result['trainable_params']:,}")
    
    # Training loop with timing
    print("\n2. Running 10 training iterations...")
    losses = []
    times = []
    gpu_stats = []
    
    for i in range(10):
        # Forward-backward
        fb_start = time.time()
        fb_result = await session.forward_backward.remote.aio(
            batch_data=TRAINING_DATA,
            loss_fn="causal_lm",
        )
        fb_time = time.time() - fb_start
        
        # Optimizer step
        opt_start = time.time()
        opt_result = await session.optim_step.remote.aio()
        opt_time = time.time() - opt_start
        
        total_time = fb_time + opt_time
        times.append(total_time)
        losses.append(fb_result["loss"])
        
        # Get GPU stats
        state = await session.get_state.remote.aio()
        if "gpu_memory_used" in state:
            gpu_stats.append(state)
        
        if i % 2 == 0:
            print(f"   Step {i+1}: loss={fb_result['loss']:.4f}, "
                  f"time={total_time:.3f}s (fb={fb_time:.3f}s, opt={opt_time:.3f}s)")
    
    avg_time = statistics.mean(times)
    avg_loss = statistics.mean(losses)
    
    print("\n✓ Completed 10 iterations")
    print(f"   Average iteration time: {avg_time:.3f}s")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   Loss improvement: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Generate samples
    print("\n3. Generating samples...")
    sample_start = time.time()
    samples = await session.sample.remote.aio(
        prompts=TEST_PROMPTS,
        max_tokens=50,
        temperature=0.7,
    )
    sample_time = time.time() - sample_start
    
    print(f"✓ Generated {len(samples['completions'])} samples in {sample_time:.2f}s")
    for i, completion in enumerate(samples['completions'][:2]):
        print(f"   [{i+1}] {completion[:80]}...")
    
    return {
        "gpu_count": 1,
        "init_time": init_time,
        "avg_iteration_time": avg_time,
        "losses": losses,
        "sample_time": sample_time,
        "gpu_stats": gpu_stats,
    }


async def test_multi_gpu(num_gpus: int = 2):
    """Test training on multiple GPUs with monitoring."""
    print("\n" + "=" * 80)
    print(f"TEST 2: MULTI-GPU TRAINING ({num_gpus} GPUs)")
    print("=" * 80)
    
    # Note: This requires deploying TrainingSession with multi-GPU support
    print(f"\n⚠️  Multi-GPU requires deploying with gpu='l40s:{num_gpus}'")
    print("    Current deployment uses single GPU only")
    print("\nTo enable multi-GPU:")
    print("  1. Update modal_runtime/training_session.py:")
    print(f"     @app.cls(gpu='l40s:{num_gpus}', ...)")
    print("  2. Add DataParallel wrapping in model_loader.py")
    print("  3. Redeploy: modal deploy modal_runtime/training_session.py")
    
    return None


def get_gpu_utilization():
    """Get current GPU utilization using pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_stats = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_stats.append({
                "gpu_id": i,
                "memory_used_gb": info.used / 1024**3,
                "memory_total_gb": info.total / 1024**3,
                "memory_percent": (info.used / info.total) * 100,
                "gpu_utilization_percent": util.gpu,
                "memory_utilization_percent": util.memory,
            })
        
        pynvml.nvmlShutdown()
        return gpu_stats
        
    except Exception as e:
        print(f"⚠️  Could not get GPU stats: {e}")
        return []


async def main():
    """Run multi-GPU tests."""
    print("=" * 80)
    print("MULTI-GPU TRAINING TEST")
    print("=" * 80)
    print(f"Model: {TEST_MODEL}")
    print(f"Training data: {len(TRAINING_DATA)} examples")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    
    # Test 1: Single GPU
    single_gpu_results = await test_single_gpu()
    
    # Test 2: Multi-GPU (requires deployment changes)
    multi_gpu_results = await test_multi_gpu(num_gpus=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if single_gpu_results:
        print("\n✓ Single GPU (L40S):")
        print(f"   Init time: {single_gpu_results['init_time']:.2f}s")
        print(f"   Avg iteration: {single_gpu_results['avg_iteration_time']:.3f}s")
        print(f"   Sample time: {single_gpu_results['sample_time']:.2f}s")
        print(f"   Loss: {single_gpu_results['losses'][0]:.4f} → {single_gpu_results['losses'][-1]:.4f}")
    
    if multi_gpu_results:
        speedup = single_gpu_results['avg_iteration_time'] / multi_gpu_results['avg_iteration_time']
        print(f"\n✓ Multi-GPU ({multi_gpu_results['gpu_count']} x L40S):")
        print(f"   Init time: {multi_gpu_results['init_time']:.2f}s")
        print(f"   Avg iteration: {multi_gpu_results['avg_iteration_time']:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Efficiency: {(speedup / multi_gpu_results['gpu_count']) * 100:.1f}%")
    else:
        print("\n⚠️  Multi-GPU test skipped (requires deployment changes)")
        print("\nTo enable multi-GPU support:")
        print("  1. See instructions above")
        print("  2. Re-run this test after deployment")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

