"""Example: Request pipelining with futures for maximum throughput.

This example demonstrates Signal's double-await pattern using _async methods
for overlapping request submission with execution, achieving maximum GPU utilization.
"""

import asyncio
import time
from rewardsignal import AsyncSignalClient
from rewardsignal.futures import FutureGroup


async def main():
    # Initialize client
    async with AsyncSignalClient(
        api_key="your-api-key-here",
        base_url="http://localhost:8000",  # Or production URL
    ) as client:
        # Create a training run
        print("Creating training run...")
        run = await client.create_run(
            base_model="meta-llama/Llama-3.2-3B",
            lora_r=32,
            lora_alpha=64,
            learning_rate=3e-4,
        )
        print(f"Created run: {run.run_id}\n")
        
        print("=== Futures Mode with Async Methods ===\n")
        
        # Example 1: Simple double-await pattern
        print("Example 1: Simple Double-Await")
        print("=" * 50)
        
        batch1 = [{"text": "Hello world"}]
        batch2 = [{"text": "Goodbye world"}]
        
        # First await: Submit request (non-blocking)
        start = time.time()
        future1 = await run.forward_backward_async(batch1, loss_fn="causal_lm")
        print(f"✓ Request 1 submitted ({(time.time() - start)*1000:.1f}ms)")
        
        future2 = await run.forward_backward_async(batch2, loss_fn="causal_lm")
        print(f"✓ Request 2 submitted ({(time.time() - start)*1000:.1f}ms)")
        
        # Second await: Wait for results (blocking)
        result1 = await future1
        print(f"✓ Request 1 completed ({(time.time() - start)*1000:.1f}ms): loss={result1['loss']:.4f}")
        
        result2 = await future2
        print(f"✓ Request 2 completed ({(time.time() - start)*1000:.1f}ms): loss={result2['loss']:.4f}")
        
        # Example 2: Pipelined training loop
        print("\n\nExample 2: Pipelined Training Loop")
        print("=" * 50)
        
        batches = [
            [{"text": f"Training example {i}"}]
            for i in range(5)
        ]
        
        start = time.time()
        
        # Submit all forward-backward requests first
        fb_futures = []
        for i, batch in enumerate(batches):
            future = await run.forward_backward_async(batch, loss_fn="causal_lm")
            fb_futures.append(future)
            print(f"✓ FB request {i+1} submitted ({(time.time() - start)*1000:.1f}ms)")
        
        # Then await results and submit optimizer steps
        opt_futures = []
        for i, fb_future in enumerate(fb_futures):
            fb_result = await fb_future
            print(f"✓ FB request {i+1} completed: loss={fb_result['loss']:.4f}")
            
            # Submit optimizer step
            opt_future = await run.optim_step_async()
            opt_futures.append(opt_future)
        
        # Wait for all optimizer steps
        for i, opt_future in enumerate(opt_futures):
            opt_result = await opt_future
            print(f"✓ Optim step {i+1} completed: step={opt_result['step']}")
        
        total_time = time.time() - start
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Average per batch: {total_time / len(batches):.2f}s")
        
        # Example 3: FutureGroup for batch operations
        print("\n\nExample 3: FutureGroup for Batch Operations")
        print("=" * 50)
        
        group = FutureGroup()
        
        # Submit multiple requests
        for i in range(3):
            batch = [{"text": f"Batch {i}"}]
            future = await client.forward_backward_async(batch, "causal_lm")
            group.add(future)
            print(f"✓ Added future {i+1} to group")
        
        # Wait for all to complete
        print("\nWaiting for all futures in group...")
        results = await group.wait_all()
        
        print(f"✓ All {len(results)} futures completed!")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: loss={result['loss']:.4f}")
        
        # Example 4: Advanced pipelining with overlapped operations
        print("\n\nExample 4: Advanced Pipelining (Overlapped FB + Optim)")
        print("=" * 50)
        
        start = time.time()
        
        # Initialize with first batch
        batch = batches[0]
        fb_future = await client.forward_backward_async(batch, "causal_lm")
        
        # Pipeline: submit next FB while waiting for prev optim
        for i in range(1, len(batches)):
            # Wait for previous FB
            fb_result = await fb_future
            print(f"Step {i}: FB complete, loss={fb_result['loss']:.4f}")
            
            # Submit next FB (overlapped with optim)
            fb_future = await client.forward_backward_async(batches[i], "causal_lm")
            
            # Submit optim step
            opt_future = await client.optim_step_async()
            opt_result = await opt_future
            print(f"Step {i}: Optim complete, step={opt_result['step']}")
        
        # Process last batch
        fb_result = await fb_future
        opt_future = await client.optim_step_async()
        opt_result = await opt_future
        print(f"Step {len(batches)}: Complete, step={opt_result['step']}")
        
        total_time = time.time() - start
        print(f"\nPipelined time: {total_time:.2f}s")
        print(f"Average per batch: {total_time / len(batches):.2f}s")
        
        # Get metrics
        metrics = client.get_metrics()
        print(f"\n=== Final Metrics ===")
        print(f"Total steps: {metrics['current_step']}")
        print(f"Avg loss: {metrics['avg_loss']:.4f}")
        print(f"Queue depth: {metrics['queue_depth']}")
        print(f"Pending requests: {metrics['pending_requests']}")
    
    print("\n✓ Futures pipelining demo complete!")


if __name__ == "__main__":
    asyncio.run(main())

