"""Quick test of stateful training session."""

import modal
import time

# Get reference to deployed class
TrainingSession = modal.Cls.lookup("signal", "TrainingSession")

# Create instance
session = TrainingSession()


print("TESTING STATEFUL TRAINING SESSION")


# Test initialize
print("\n1. Testing initialize...")
start = time.time()
result = session.initialize.remote(
    user_id="test_user",
    run_id="test_run_stateful",
    base_model="HuggingFaceTB/SmolLM2-135M",
    lora_r=8,
    max_seq_length=512,
)
init_time = time.time() - start
print(f"✓ Initialize completed in {init_time:.2f}s")
print(f"   Status: {result['status']}")
print(f"   Current step: {result['current_step']}")
print(f"   Trainable params: {result['trainable_params']:,}")

# Test forward_backward (first call)
print("\n2. Testing forward_backward (first call)...")
batch = [{"text": "The quick brown fox jumps over the lazy dog"}]
start = time.time()
fb_result = session.forward_backward.remote(
    batch_data=batch,
    loss_fn="causal_lm",
)
fb1_time = time.time() - start
print(f"✓ Forward-backward completed in {fb1_time:.2f}s")
print(f"   Loss: {fb_result['loss']:.4f}")
print(f"   Step: {fb_result['step']}")
print(f"   Grad norm: {fb_result.get('grad_norm', 0):.4f}")

# Test optim_step
print("\n3. Testing optim_step...")
start = time.time()
opt_result = session.optim_step.remote()
opt_time = time.time() - start
print(f"✓ Optimizer step completed in {opt_time:.2f}s")
print(f"   New step: {opt_result['step']}")
print(f"   Learning rate: {opt_result['learning_rate']}")

# Test second forward_backward (should be FAST!)
print("\n4. Testing forward_backward (second call - should be fast!)...")
start = time.time()
fb_result2 = session.forward_backward.remote(
    batch_data=batch,
    loss_fn="causal_lm",
)
fb2_time = time.time() - start
print(f"✓ Forward-backward completed in {fb2_time:.2f}s")
print(f"   Loss: {fb_result2['loss']:.4f}")
print(f"   Step: {fb_result2['step']}")

# Test sample
print("\n5. Testing sample generation...")
start = time.time()
sample_result = session.sample.remote(
    prompts=["The quick brown"],
    max_tokens=20,
    temperature=0.7,
)
sample_time = time.time() - start
print(f"✓ Sample generation completed in {sample_time:.2f}s")
print(f"   Generated: {sample_result['outputs'][0][:100]}...")

# Test get_state
print("\n6. Testing get_state...")
state = session.get_state.remote()
print("✓ State retrieved")
print(f"   Status: {state['status']}")
print(f"   Current step: {state['current_step']}")
print(f"   Last checkpoint step: {state['last_checkpoint_step']}")

# Performance summary

print("PERFORMANCE SUMMARY")

print(f"Initialize (cold start):     {init_time:.2f}s")
print(f"Forward-backward (1st):      {fb1_time:.2f}s")
print(f"Optimizer step:              {opt_time:.2f}s")
print(f"Forward-backward (2nd):      {fb2_time:.2f}s  ⚡ WARM CALL")
print(f"Sample generation:           {sample_time:.2f}s")
print(f"\nTotal time (init + 1 iter):  {init_time + fb1_time + opt_time:.2f}s")
print(f"Warm iteration time:         {fb2_time + opt_time:.2f}s")

if fb2_time < 10:
    print("\n✅ SUCCESS: Warm calls are fast (<10s)!")
else:
    print(f"\n⚠️  WARNING: Warm call took {fb2_time:.2f}s (expected <10s)")

print("\n✅ All tests passed!")
