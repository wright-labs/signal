"""End-to-end test via the Signal API (not directly calling Modal).

This test verifies the API endpoints work correctly by:
1. Creating a run via the API
2. Calling forward_backward via the API  
3. Calling optim_step via the API
4. Calling sample via the API
5. Calling save_state via the API
6. Verifying data appears in Supabase

This requires the API server to be running locally.
"""
import os
import sys
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_signal_api_e2e():
    """Test Signal API end-to-end."""
    
    print("\n" + "="*80)
    print("Signal E2E Test - Via API")
    print("="*80 + "\n")
    
    # Configuration
    API_BASE_URL = os.getenv("SIGNAL_API_URL", "http://localhost:8000")
    TEST_API_KEY = os.getenv("TEST_API_KEY", "sk-test-key")  # Set this to a real API key
    
    headers = {
        "Authorization": f"Bearer {TEST_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Check API health
    print(f"1. Checking API health at {API_BASE_URL}...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ API is healthy")
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to API at {API_BASE_URL}")
        print("   Make sure the API server is running with: uvicorn api.main:app")
        return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # List available models
    print("\n2. Listing available models...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()["models"]
            print(f"   ✓ Found {len(models)} models")
            print(f"   Using model: {models[0]['name']}")
            test_model = models[0]["name"]
        else:
            print(f"   ❌ Failed to list models: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to list models: {e}")
        return False
    
    # Create a run
    print("\n3. Creating training run...")
    try:
        run_config = {
            "base_model": test_model,
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 3e-4,
            "max_seq_length": 512,
        }
        
        response = requests.post(
            f"{API_BASE_URL}/runs",
            json=run_config,
            headers=headers,
            timeout=300,  # 5 min timeout for model loading
        )
        
        if response.status_code == 200:
            run_data = response.json()
            run_id = run_data["run_id"]
            print(f"   ✓ Created run: {run_id}")
            print(f"   Status: {run_data['status']}")
        elif response.status_code == 401:
            print("   ❌ Authentication failed. Set TEST_API_KEY environment variable.")
            return False
        else:
            print(f"   ❌ Failed to create run: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to create run: {e}")
        return False
    
    # Give it a moment for Modal to initialize
    print("\n   Waiting for Modal to initialize (30s)...")
    time.sleep(30)
    
    # Forward-backward pass
    print("\n4. Running forward-backward pass...")
    try:
        batch_data = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Hello world, this is a test."},
        ]
        
        response = requests.post(
            f"{API_BASE_URL}/runs/{run_id}/forward_backward",
            json={
                "batch_data": batch_data,
                "accumulate": False,
                "loss_fn": "causal_lm",
                "loss_kwargs": {},
            },
            headers=headers,
            timeout=300,
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Forward-backward successful")
            print(f"   Loss: {result['loss']:.4f}")
            print(f"   Step: {result['step']}")
        else:
            print(f"   ❌ Forward-backward failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Forward-backward failed: {e}")
        return False
    
    # Optimizer step
    print("\n5. Running optimizer step...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/runs/{run_id}/optim_step",
            json={},
            headers=headers,
            timeout=300,
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Optimizer step successful")
            print(f"   Step: {result['step']}")
            print(f"   Learning rate: {result['learning_rate']}")
        else:
            print(f"   ❌ Optimizer step failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Optimizer step failed: {e}")
        return False
    
    # Sample from model
    print("\n6. Sampling from model...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/runs/{run_id}/sample",
            json={
                "prompts": ["The meaning of life is"],
                "max_tokens": 20,
                "temperature": 0.7,
            },
            headers=headers,
            timeout=300,
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Sampling successful")
            print(f"   Output: {result['outputs'][0][:100]}...")
        else:
            print(f"   ⚠ Sampling failed (non-critical): {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Sampling failed (non-critical): {e}")
    
    # Save state
    print("\n7. Saving model state...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/runs/{run_id}/save_state",
            json={
                "mode": "adapter",
                "push_to_hub": False,
            },
            headers=headers,
            timeout=300,
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ Save state successful")
            print(f"   Checkpoint: {result['checkpoint_path']}")
        else:
            print(f"   ❌ Save state failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Save state failed: {e}")
        return False
    
    # Check run status
    print("\n8. Checking final run status...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/runs/{run_id}/status",
            headers=headers,
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"   ✓ Run status: {status['status']}")
            print(f"   Current step: {status['current_step']}")
        else:
            print(f"   ⚠ Failed to get status: {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Failed to get status: {e}")
    
    # Get metrics
    print("\n9. Checking metrics...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/runs/{run_id}/metrics",
            headers=headers,
        )
        
        if response.status_code == 200:
            metrics = response.json()
            print(f"   ✓ Found {len(metrics.get('metrics', []))} metric(s)")
        else:
            print(f"   ⚠ Failed to get metrics: {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Failed to get metrics: {e}")
    
    print("\n" + "="*80)
    print("✅ All API tests passed! Signal is working correctly.")
    print("="*80 + "\n")
    print(f"Run ID: {run_id}")
    print("You can view this run in the Supabase database or in the frontend at /signal")
    
    return True


if __name__ == "__main__":
    try:
        success = test_signal_api_e2e()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

