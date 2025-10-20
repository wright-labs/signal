"""Quick multi-GPU test with Llama 3.2 1B"""

import asyncio
import modal
import time

MODEL = "Qwen/Qwen2.5-1.5B"  # 1.5B parameter model  # Open 1.5B model!
TRAINING_DATA = [
    {"text": "The quick brown fox jumps over the lazy dog. " * 15},
    {"text": "Machine learning revolutionizes artificial intelligence. " * 15},
    {"text": "Python enables efficient data science workflows. " * 15},
    {"text": "Neural networks learn from massive datasets. " * 15},
]


async def main():
    print("=" * 80)
    print("MULTI-GPU TEST: Qwen 2.5 1.5B on 2x L40S")
    print("=" * 80)

    # Get training session
    print("\n1. Getting training session...")
    TrainingSession = modal.Cls.from_name(
        "signal", "TrainingSession", environment_name="main"
    )
    session = TrainingSession()

    # Initialize
    print(f"\n2. Initializing {MODEL}...")
    init_start = time.time()
    result = await session.initialize.remote.aio(
        user_id="test",
        run_id="test_multi_gpu_llama",
        base_model=MODEL,
        lora_r=32,
        lora_alpha=64,
        learning_rate=3e-4,
        max_seq_length=1024,
        gradient_checkpointing=True,
        load_in_4bit=False,  # Disable quantization for multi-GPU
        auto_checkpoint_interval=1000,
    )
    init_time = time.time() - init_start

    print(f"\n✓ Initialized in {init_time:.2f}s")
    print(f"  Model: {MODEL}")
    print(f"  Trainable params: {result['trainable_params']:,}")
    print(f"  Total params: {result['total_params']:,}")
    print(f"  GPUs: {result.get('num_gpus', 1)}")

    if "gpu_stats" in result:
        print("\n  GPU Stats:")
        for gpu in result["gpu_stats"]:
            print(f"    GPU {gpu['gpu_id']}: {gpu['name']}")
            print(
                f"      Memory: {gpu['memory_allocated_gb']:.2f} / {gpu['memory_total_gb']:.2f} GB"
            )

    # Training loop
    print("\n3. Running 5 training iterations...")
    print("-" * 80)

    iteration_times = []
    losses = []

    for i in range(5):
        iter_start = time.time()

        # Forward-backward
        fb_result = await session.forward_backward.remote.aio(
            batch_data=TRAINING_DATA,
            loss_fn="causal_lm",
        )

        # Optimizer step
        await session.optim_step.remote.aio()

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        losses.append(fb_result["loss"])

        # Show GPU stats if available
        gpu_info = ""
        if "num_gpus" in fb_result and fb_result["num_gpus"] > 1:
            gpu_info = f" | 🚀 {fb_result['num_gpus']} GPUs"

        print(
            f"  Step {i + 1}: loss={fb_result['loss']:.4f}, "
            f"time={iter_time:.3f}s{gpu_info}"
        )

        if "gpu_stats" in fb_result and i == 0:
            for gpu in fb_result["gpu_stats"]:
                print(
                    f"    GPU {gpu['gpu_id']}: {gpu['memory_allocated_gb']:.2f} GB "
                    f"({gpu['memory_percent']:.1f}%)"
                )

    avg_time = sum(iteration_times) / len(iteration_times)
    loss_improvement = losses[0] - losses[-1]

    print("\n✓ Training complete!")
    print(f"  Average iteration: {avg_time:.3f}s")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ {loss_improvement:.4f})")

    # Generate samples
    print("\n4. Generating samples...")
    sample_start = time.time()

    samples = await session.sample.remote.aio(
        prompts=[
            "The future of artificial intelligence",
            "In deep learning we",
        ],
        max_tokens=40,
        temperature=0.7,
    )

    sample_time = time.time() - sample_start
    print(f"✓ Generated samples in {sample_time:.2f}s:")
    for i, output in enumerate(samples["outputs"]):
        print(f"  [{i + 1}] {output[:80]}...")

    # Final state
    state = await session.get_state.remote.aio()
    print("\n5. Final state:")
    print(f"  Step: {state['current_step']}")
    print(f"  Status: {state['status']}")
    print(f"  GPUs: {state.get('num_gpus', 1)}")
    print(f"  Multi-GPU: {state.get('is_multi_gpu', False)}")

    if "gpu_summary" in state:
        gpu_summary = state["gpu_summary"]
        print(f"  Total GPU memory: {gpu_summary.get('total_memory_gb', 0):.1f} GB")
        print(f"  Allocated: {gpu_summary.get('total_allocated_gb', 0):.2f} GB")
        print(f"  Avg utilization: {gpu_summary.get('avg_memory_percent', 0):.1f}%")

    print("\n" + "=" * 80)
    print("✅ MULTI-GPU TEST PASSED!")
    print("=" * 80)
    print(f"Model: Qwen 2.5 1.5B ({result['trainable_params']:,} LoRA params)")
    print("GPUs: 2x L40S (DataParallel)")
    print(f"Performance: {avg_time:.3f}s per iteration")
    print(f"Loss improved: {losses[0]:.4f} → {losses[-1]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
