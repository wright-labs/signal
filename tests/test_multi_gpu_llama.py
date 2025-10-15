"""Test multi-GPU training with Llama 3.2 or Qwen models.

This script tests the full multi-GPU pipeline:
1. Deploy multi-GPU training sessions
2. Train on larger models (Llama 3.2 1B/3B or Qwen)
3. Monitor per-GPU utilization
4. Compare single vs multi-GPU performance
"""
import asyncio
import time
from rewardsignal import SignalClient
import os

# Test configurations
MODELS_TO_TEST = [
    {
        "name": "Llama 3.2 1B",
        "model_id": "meta-llama/Llama-3.2-1B",
        "lora_r": 32,
        "max_seq_length": 2048,
    },
    {
        "name": "Llama 3.2 3B",
        "model_id": "meta-llama/Llama-3.2-3B",
        "lora_r": 64,
        "max_seq_length": 2048,
    },
    {
        "name": "Qwen 2.5 1.5B",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "lora_r": 32,
        "max_seq_length": 2048,
    },
]

GPU_CONFIGS_TO_TEST = [
    "l40s:1",   # Single GPU baseline
    "l40s:2",   # 2x GPUs
    # "l40s:4", # 4x GPUs (expensive, optional)
]

# Training data
TRAINING_DATA = [
    {"text": "The quick brown fox jumps over the lazy dog. " * 20},
    {"text": "Machine learning is revolutionizing artificial intelligence. " * 20},
    {"text": "Python is widely used for data science and AI applications. " * 20},
    {"text": "Deep learning models require substantial computational resources. " * 20},
]

EVAL_PROMPTS = [
    "The future of AI is",
    "In machine learning",
    "To train a model",
]


async def test_model_gpu_combo(model_config: dict, gpu_config: str, api_key: str):
    """Test a specific model with a specific GPU configuration."""
    print("\n" + "=" * 80)
    print(f"TEST: {model_config['name']} on {gpu_config}")
    print("=" * 80)
    
    client = SignalClient(
        api_key=api_key,
        base_url=os.getenv("API_URL", "http://localhost:8000")
    )
    
    try:
        # Create run with specific GPU config
        print(f"\n1. Creating run with {gpu_config}...")
        create_start = time.time()
        
        run = client.create_run(
            base_model=model_config["model_id"],
            gpu_config=gpu_config,  # ← Specify GPU config!
            lora_r=model_config["lora_r"],
            lora_alpha=model_config["lora_r"] * 2,
            max_seq_length=model_config["max_seq_length"],
            gradient_checkpointing=True,
        )
        
        create_time = time.time() - create_start
        print(f"✓ Run created in {create_time:.2f}s")
        print(f"  Run ID: {run.run_id}")
        print(f"  Model: {model_config['model_id']}")
        print(f"  GPU: {gpu_config}")
        
        # Get session state to see GPU info
        state = run.get_state()
        print(f"\n2. Session state:")
        print(f"  Status: {state.get('status')}")
        print(f"  GPUs: {state.get('num_gpus', 1)}")
        print(f"  Multi-GPU: {state.get('is_multi_gpu', False)}")
        
        if "gpu_summary" in state:
            gpu_summary = state["gpu_summary"]
            print(f"  Total GPU memory: {gpu_summary.get('total_memory_gb', 0):.1f} GB")
            print(f"  Allocated memory: {gpu_summary.get('total_allocated_gb', 0):.1f} GB")
        
        # Training loop
        print(f"\n3. Running 5 training iterations...")
        iteration_times = []
        losses = []
        
        for i in range(5):
            iter_start = time.time()
            
            # Forward-backward
            fb_result = run.forward_backward(TRAINING_DATA)
            
            # Optimizer step
            opt_result = run.optim_step()
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            losses.append(fb_result.loss)
            
            # Print GPU stats if available
            gpu_info = ""
            if hasattr(fb_result, 'num_gpus') and fb_result.num_gpus:
                gpu_info = f" | GPUs: {fb_result.num_gpus}"
            
            print(f"  Step {i+1}: loss={fb_result.loss:.4f}, "
                  f"time={iter_time:.3f}s{gpu_info}")
        
        avg_time = sum(iteration_times) / len(iteration_times)
        loss_improvement = losses[0] - losses[-1]
        
        print(f"\n✓ Training complete")
        print(f"  Average iteration time: {avg_time:.3f}s")
        print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ {loss_improvement:.4f})")
        
        # Generate samples
        print(f"\n4. Generating samples...")
        sample_start = time.time()
        
        samples = run.sample(
            prompts=EVAL_PROMPTS,
            max_tokens=30,
            temperature=0.7,
        )
        
        sample_time = time.time() - sample_start
        print(f"✓ Generated {len(samples.outputs)} samples in {sample_time:.2f}s")
        for i, output in enumerate(samples.outputs[:2]):
            print(f"  [{i+1}] {output[:70]}...")
        
        # Final state with GPU stats
        final_state = run.get_state()
        
        print(f"\n5. Final state:")
        print(f"  Step: {final_state.get('current_step', 0)}")
        print(f"  Status: {final_state.get('status')}")
        
        if "gpu_summary" in final_state:
            gpu_summary = final_state["gpu_summary"]
            print(f"  GPU memory used: {gpu_summary.get('total_allocated_gb', 0):.2f} GB")
            if "per_gpu" in gpu_summary:
                for gpu in gpu_summary["per_gpu"]:
                    print(f"    GPU {gpu['gpu_id']}: {gpu['memory_allocated_gb']:.2f} GB "
                          f"({gpu['memory_percent']:.1f}%)")
        
        return {
            "model": model_config["name"],
            "gpu_config": gpu_config,
            "create_time": create_time,
            "avg_iteration_time": avg_time,
            "sample_time": sample_time,
            "loss_improvement": loss_improvement,
            "final_state": final_state,
        }
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run comprehensive multi-GPU tests."""
    print("=" * 80)
    print("MULTI-GPU TRAINING TEST")
    print("=" * 80)
    
    # Get API key
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ API_KEY environment variable not set")
        return
    
    print(f"API URL: {os.getenv('API_URL', 'http://localhost:8000')}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    print(f"GPU configs: {GPU_CONFIGS_TO_TEST}")
    
    # Run tests
    results = []
    
    for model_config in MODELS_TO_TEST:
        for gpu_config in GPU_CONFIGS_TO_TEST:
            result = await test_model_gpu_combo(model_config, gpu_config, api_key)
            if result:
                results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(2)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not results:
        print("❌ No successful tests")
        return
    
    # Group by model
    by_model = {}
    for r in results:
        if r["model"] not in by_model:
            by_model[r["model"]] = []
        by_model[r["model"]].append(r)
    
    for model_name, model_results in by_model.items():
        print(f"\n{model_name}:")
        
        # Find single GPU baseline
        baseline = next((r for r in model_results if ":1" in r["gpu_config"]), None)
        
        for r in sorted(model_results, key=lambda x: x["gpu_config"]):
            speedup = ""
            if baseline and r != baseline:
                speedup_factor = baseline["avg_iteration_time"] / r["avg_iteration_time"]
                speedup = f" (Speedup: {speedup_factor:.2f}x)"
            
            print(f"  {r['gpu_config']:10s}: "
                  f"{r['avg_iteration_time']:.3f}s/iter | "
                  f"Loss Δ: {r['loss_improvement']:.4f}{speedup}")
    
    # Cost analysis
    print(f"\n" + "-" * 80)
    print("Cost Analysis (estimated for 1000 steps):")
    print("-" * 80)
    
    GPU_COSTS = {
        "l40s:1": 1.60,
        "l40s:2": 3.20,
        "l40s:4": 6.40,
    }
    
    for model_name, model_results in by_model.items():
        print(f"\n{model_name}:")
        for r in sorted(model_results, key=lambda x: x["gpu_config"]):
            time_hours = (r["avg_iteration_time"] * 1000) / 3600
            cost = time_hours * GPU_COSTS.get(r["gpu_config"], 0)
            print(f"  {r['gpu_config']:10s}: "
                  f"{time_hours:.2f}h = ${cost:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ Multi-GPU testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

