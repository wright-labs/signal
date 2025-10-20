#!/usr/bin/env python3
"""Real-world test of Signal SDK with single and multi-GPU training."""

import sys
import time
from pathlib import Path

# Add client to path for local development
sys.path.insert(0, str(Path(__file__).parent / "client"))

from rewardsignal import SignalClient


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_single_gpu_test(client: SignalClient):
    """Test single GPU training with Llama on H100."""
    print_section("TEST 1: Single GPU Training (Llama 3.2-1B)")

    # Create run
    print("📦 Creating training run...")
    print("   Model: meta-llama/Llama-3.2-1B")
    print("   Duration: ~1 minute")

    start = time.time()

    try:
        run = client.create_run(
            base_model="meta-llama/Llama-3.2-1B",
            lora_r=8,
            lora_alpha=16,
            learning_rate=5e-4,
            max_seq_length=512,  # Shorter for faster testing
            bf16=True,
            gradient_checkpointing=True,
        )

        elapsed = time.time() - start
        print(f"✅ Run created in {elapsed:.1f}s")
        print(f"   Run ID: {run.run_id}")

    except Exception as e:
        print(f"❌ Failed to create run: {e}")
        return None

    # Prepare training batch
    batch = [
        {"text": "The future of AI is decentralized and open source."},
        {"text": "Machine learning models are becoming more efficient every day."},
        {"text": "Language models can understand context and generate coherent text."},
        {"text": "Deep learning has revolutionized natural language processing."},
    ]

    # Run training for ~1 minute
    print("\n🚀 Starting training loop...")
    num_steps = 5  # Small number for quick test

    try:
        for step in range(num_steps):
            step_start = time.time()

            # Forward-backward pass
            print(f"\n   Step {step + 1}/{num_steps}")
            fb_result = run.forward_backward(batch=batch, loss_fn="causal_lm")

            loss = fb_result["loss"]
            grad_norm = fb_result.get("grad_norm", "N/A")
            print(f"      Loss: {loss:.4f}")
            print(f"      Grad norm: {grad_norm}")

            # Optimizer step
            opt_result = run.optim_step()

            step_elapsed = time.time() - step_start
            print(f"      Time: {step_elapsed:.1f}s")

            # Sample every 2 steps
            if (step + 1) % 2 == 0:
                print("\n   🎲 Generating sample...")
                try:
                    samples = run.sample(
                        prompts=["The future of AI"],
                        max_tokens=30,
                        temperature=0.8,
                    )
                    print(f"      → {samples['outputs'][0]}")
                except Exception as e:
                    print(f"      Warning: Sampling failed: {e}")

        # Get final status
        print("\n📊 Final status:")
        status = run.get_status()
        print(f"   Status: {status.get('status', 'unknown')}")
        print(f"   Steps completed: {status.get('current_step', 'unknown')}")

        # Save adapter
        print("\n💾 Saving adapter...")
        save_start = time.time()
        artifact = run.save_state(mode="adapter", push_to_hub=False)
        save_elapsed = time.time() - save_start

        print(f"✅ Adapter saved in {save_elapsed:.1f}s")
        print(f"   Path: {artifact.get('checkpoint_path', 'unknown')}")

        total_time = time.time() - start
        print(f"\n✅ Single GPU test completed in {total_time:.1f}s")

        return run

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_multi_gpu_test(client: SignalClient):
    """Test multi-GPU training with Qwen."""
    print_section("TEST 2: Qwen 2.5-7B Fine-tuning")

    print("📦 Creating training run...")
    print("   Model: Qwen/Qwen2.5-7B-Instruct")
    print("   Duration: ~2 minutes")

    start = time.time()

    try:
        run = client.create_run(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            lora_r=16,
            lora_alpha=32,
            learning_rate=3e-4,
            max_seq_length=1024,
            bf16=True,
            gradient_checkpointing=True,
        )

        elapsed = time.time() - start
        print(f"✅ Run created in {elapsed:.1f}s")
        print(f"   Run ID: {run.run_id}")

    except Exception as e:
        print(f"❌ Failed to create run: {e}")
        return None

    # Prepare chat-style training batch (Qwen is instruction-tuned)
    batch = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain neural networks simply."},
                {
                    "role": "assistant",
                    "content": "Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes (neurons) that process information in layers, learning patterns from examples to make predictions or decisions.",
                },
            ]
        },
    ]

    # Run training
    print("\n🚀 Starting multi-GPU training loop...")
    num_steps = 3  # Smaller for multi-GPU test

    try:
        for step in range(num_steps):
            step_start = time.time()

            print(f"\n   Step {step + 1}/{num_steps}")

            # Forward-backward with chat format
            fb_result = run.forward_backward(batch=batch, loss_fn="causal_lm")

            loss = fb_result["loss"]
            grad_norm = fb_result.get("grad_norm", "N/A")
            print(f"      Loss: {loss:.4f}")
            print(f"      Grad norm: {grad_norm}")

            # Optimizer step
            opt_result = run.optim_step()

            step_elapsed = time.time() - step_start
            print(f"      Time: {step_elapsed:.1f}s")

        # Sample from fine-tuned model
        print("\n🎲 Generating completion from fine-tuned model...")
        try:
            samples = run.sample(
                prompts=["Explain quantum computing in simple terms:"],
                max_tokens=50,
                temperature=0.7,
            )
            print(f"   → {samples['outputs'][0]}")
        except Exception as e:
            print(f"   Warning: Sampling failed: {e}")

        # Get status
        print("\n📊 Final status:")
        status = run.get_status()
        print(f"   Status: {status.get('status', 'unknown')}")
        print(f"   Steps completed: {status.get('current_step', 'unknown')}")

        # Save merged model
        print("\n💾 Saving merged model...")
        save_start = time.time()
        artifact = run.save_state(mode="merged", push_to_hub=False)
        save_elapsed = time.time() - save_start

        print(f"✅ Merged model saved in {save_elapsed:.1f}s")
        print(f"   Path: {artifact.get('checkpoint_path', 'unknown')}")

        total_time = time.time() - start
        print(f"\n✅ Multi-GPU test completed in {total_time:.1f}s")

        return run

    except Exception as e:
        print(f"\n❌ Multi-GPU training failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print_section("🧪 Signal SDK Integration Tests")

    # Load API key from .env
    env_path = Path(__file__).parent / ".env"
    api_key = None

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("TEST_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        print("❌ No TEST_API_KEY found in .env!")
        print("   Please add TEST_API_KEY=sk-... to signal/.env")
        sys.exit(1)

    print(f"✅ API key loaded: {api_key[:20]}...")

    # Initialize client
    print("\n🔌 Connecting to Signal API...")
    client = SignalClient(
        api_key=api_key,
        base_url="http://localhost:8000",
        timeout=600,  # 10 minute timeout for GPU operations
    )

    # Test health
    try:
        import requests

        resp = requests.get(f"{client.base_url}/health", timeout=5)
        if resp.status_code == 200:
            print("✅ API is healthy")
        else:
            print(f"⚠️  API returned {resp.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        print("   Make sure the API server is running: ./dev_server.sh")
        sys.exit(1)

    # List available models
    print("\n📋 Available models:")
    try:
        models_response = client.list_models()
        models = (
            models_response.get("models", [])
            if isinstance(models_response, dict)
            else models_response
        )
        for i, model in enumerate(models[:5], 1):
            print(f"   {i}. {model}")
        if len(models) > 5:
            print(f"   ... and {len(models) - 5} more")
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")

    # Run tests
    overall_start = time.time()

    # Test 1: Single GPU
    single_result = run_single_gpu_test(client)

    # Test 2: Multi-GPU
    multi_result = run_multi_gpu_test(client)

    # Summary
    overall_elapsed = time.time() - overall_start

    print_section("📈 Test Summary")
    print(f"Llama 3.2-1B Test:       {'✅ PASS' if single_result else '❌ FAIL'}")
    print(f"Qwen 2.5-7B Test:        {'✅ PASS' if multi_result else '❌ FAIL'}")
    print(
        f"\nTotal test time: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} minutes)"
    )

    if single_result and multi_result:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests failed - check logs above")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
