#!/usr/bin/env python3
"""Quick test of rewardsignal SDK."""

import os
from dotenv import load_dotenv
from rewardsignal import SignalClient

# Load API key
load_dotenv()
api_key = os.getenv("TEST_API_KEY")

# Initialize client
client = SignalClient(api_key=api_key, base_url="http://localhost:8000")

print("🚀 Creating training run...")
run = client.create_run(base_model="Qwen/Qwen2.5-3B", lora_r=8)
print(f"✅ Run created: {run.run_id}")

# Training data
batch = [{"text": "The quick brown fox jumps over the lazy dog."}]

print("\n📊 Training for 3 steps...")
for step in range(3):
    result = run.forward_backward(batch=batch)
    run.optim_step()
    print(f"  Step {step + 1}: Loss = {result['loss']:.4f}")

print("\n🎉 Training completed successfully!")

