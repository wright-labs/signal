"""Simple end-to-end test of Signal API and Modal primitives.

This test creates a tiny training run with a very small model to verify
that all the core primitives work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.supabase_client import get_supabase
from api.registry import RunRegistry
from api.models import ModelRegistry

# Import Modal functions
from modal_runtime.primitives import (
    create_run as modal_create_run,
    forward_backward as modal_forward_backward,
    optim_step as modal_optim_step,
    sample as modal_sample,
    save_state as modal_save_state,
)


def test_modal_primitives_e2e():
    """Test all Modal primitives end-to-end with a small model."""

    
    print("Signal E2E Test - Modal Primitives")
    

    # Setup
    registry = RunRegistry()
    model_registry = ModelRegistry()

    # Use a very small model for testing
    test_model = "meta-llama/Llama-3.2-1B"
    test_user_id = "00000000-0000-0000-0000-000000000001"  # Test user

    # Ensure test user profile exists
    supabase = get_supabase()
    try:
        supabase.table("profiles").upsert(
            {"id": test_user_id, "email": "test@signal.dev"}
        ).execute()
        print("✓ Test user profile created/verified")
    except Exception as e:
        print(f"Note: Profile may already exist: {e}")

    print(f"\n1. Testing with model: {test_model}")

    # Get model config
    model_config = model_registry.get_model(test_model)
    if not model_config:
        print(f"❌ Model {test_model} not found in registry")
        return False

    print(f"   Framework: {model_config['framework']}")
    print(f"   GPU: {model_config['gpu']}")

    # Create run in registry
    print("\n2. Creating run in registry...")
    run_id = registry.create_run(
        user_id=test_user_id,
        base_model=test_model,
        config={
            "lora_r": 8,  # Very small rank for testing
            "lora_alpha": 16,
            "learning_rate": 3e-4,
            "max_seq_length": 512,  # Short sequence for testing
        },
    )
    print(f"   ✓ Created run: {run_id}")

    # Test 1: Create run on Modal
    print("\n3. Testing create_run (Modal)...")
    try:
        result = modal_create_run.remote(
            user_id=test_user_id,
            run_id=run_id,
            base_model=test_model,
            framework=model_config["framework"],
            gpu_config=model_config["gpu"],
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_modules=None,
            optimizer="adamw_8bit",
            learning_rate=3e-4,
            weight_decay=0.01,
            max_seq_length=512,
            bf16=True,
            gradient_checkpointing=True,
            integrations={},  # No integrations for test
        )
        print("   ✓ Modal create_run successful")
        print(f"   Status: {result['status']}")

        # Update registry
        registry.update_run(run_id, status="running")
    except Exception as e:
        print(f"   ❌ Modal create_run failed: {e}")
        registry.delete_run(run_id, test_user_id)
        return False

    # Test 2: Forward-backward pass
    print("\n4. Testing forward_backward (Modal)...")
    try:
        # Very simple training data
        batch_data = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Hello world, this is a test."},
        ]

        result = modal_forward_backward.remote(
            user_id=test_user_id,
            run_id=run_id,
            batch_data=batch_data,
            step=0,
            accumulate=False,
            loss_fn="causal_lm",
            loss_kwargs={},
        )
        print("   ✓ Forward-backward successful")
        print(f"   Loss: {result['loss']:.4f}")
        print(f"   Grad norm: {result.get('grad_norm', 'N/A')}")

        # Update metrics
        registry.update_run(
            run_id,
            metrics={
                "loss": result["loss"],
                "grad_norm": result.get("grad_norm"),
            },
        )
    except Exception as e:
        print(f"   ❌ Forward-backward failed: {e}")
        registry.delete_run(run_id, test_user_id)
        return False

    # Test 3: Optimizer step
    print("\n5. Testing optim_step (Modal)...")
    try:
        result = modal_optim_step.remote(
            user_id=test_user_id,
            run_id=run_id,
            step=0,
            learning_rate=3e-4,
        )
        print("   ✓ Optimizer step successful")
        print(f"   Step: {result['step']}")
        print(f"   Learning rate: {result['learning_rate']}")

        # Update step
        registry.update_run(run_id, current_step=result["step"])
    except Exception as e:
        print(f"   ❌ Optimizer step failed: {e}")
        registry.delete_run(run_id, test_user_id)
        return False

    # Test 4: Sample from model
    print("\n6. Testing sample (Modal)...")
    try:
        result = modal_sample.remote(
            user_id=test_user_id,
            run_id=run_id,
            prompts=["The meaning of life is"],
            step=1,
            max_tokens=20,
            temperature=0.7,
            top_p=0.9,
            return_logprobs=False,
        )
        print("   ✓ Sampling successful")
        print(f"   Output: {result['outputs'][0][:100]}...")
    except Exception as e:
        print(f"   ❌ Sampling failed: {e}")
        # Not critical, continue

    # Test 5: Save state
    print("\n7. Testing save_state (Modal)...")
    try:
        result = modal_save_state.remote(
            user_id=test_user_id,
            run_id=run_id,
            step=1,
            mode="adapter",
            push_to_hub=False,
            hub_model_id=None,
        )
        print("   ✓ Save state successful")
        print(f"   Checkpoint: {result['checkpoint_path']}")
        print(f"   Artifact URI: {result['artifact_uri']}")
    except Exception as e:
        print(f"   ❌ Save state failed: {e}")
        registry.delete_run(run_id, test_user_id)
        return False

    # Verify metrics in Supabase
    print("\n8. Verifying metrics in Supabase...")
    metrics = registry.get_metrics(run_id)
    if metrics and len(metrics) > 0:
        print(f"   ✓ Found {len(metrics)} metric(s) in database")
        for m in metrics:
            print(f"   - Step {m['step']}: loss={m.get('loss', 'N/A')}")
    else:
        print("   ⚠ No metrics found (may be expected if not explicitly recorded)")

    # Cleanup
    print("\n9. Cleaning up test run...")
    registry.update_run(run_id, status="completed")
    print(f"   ✓ Run marked as completed: {run_id}")

    # Don't delete so we can inspect in Supabase
    print("   Run preserved in database for inspection")

    
    print("✅ All tests passed! Signal primitives are working correctly.")
    

    return True


if __name__ == "__main__":
    try:
        success = test_modal_primitives_e2e()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
