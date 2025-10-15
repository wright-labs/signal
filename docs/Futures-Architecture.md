# Futures Architecture & Request Pipelining

Signal implements a Tinker-style futures architecture that enables request pipelining for maximum throughput and GPU utilization.

## Overview

The **double-await pattern** separates request submission from execution:

1. **First await**: Submits request to queue (non-blocking, ordering guaranteed)
2. **Second await**: Waits for computation to complete (blocking, returns result)

This allows you to submit multiple requests while previous ones execute, maximizing GPU utilization.

## Enabling Futures Mode

Futures mode is controlled by environment variable:

```bash
export SIGNAL_ENABLE_FUTURES=true
```

Or in Python:

```python
import os
os.environ["SIGNAL_ENABLE_FUTURES"] = "true"

from rewardsignal.async_training_client_v2 import AsyncTrainingClientV2

client = AsyncTrainingClientV2(
    run_id="your-run-id",
    api_key="your-api-key",
    enable_futures=True,  # Explicit override
)
```

## Basic Usage

### Simple Double-Await

```python
# First await: Submit request (non-blocking)
future = await client.forward_backward_async(batch, "ppo")

# Can submit more requests here while first executes
future2 = await client.forward_backward_async(batch2, "ppo")

# Second await: Wait for completion (blocking)
result1 = await future
result2 = await future2
```

### Pipelined Training Loop

```python
# Submit multiple forward-backward requests
fb_futures = []
for batch in dataloader:
    future = await client.forward_backward_async(batch, "causal_lm")
    fb_futures.append(future)

# Then await results and submit optimizer steps
for fb_future in fb_futures:
    result = await fb_future
    await client.optim_step_async()
```

## Advanced Patterns

### FutureGroup for Batch Operations

```python
from rewardsignal.futures import FutureGroup

group = FutureGroup()

# Submit multiple requests
for batch in batches:
    future = await client.forward_backward_async(batch, "ppo")
    group.add(future)

# Wait for all to complete
results = await group.wait_all()

# Or wait for first to complete
first_result = await group.wait_any()
```

### Overlapped Forward-Backward and Optimizer Steps

```python
# Initialize with first batch
fb_future = await client.forward_backward_async(batches[0], "ppo")

# Pipeline: submit next FB while waiting for prev optim
for i in range(1, len(batches)):
    # Wait for previous FB
    fb_result = await fb_future
    
    # Submit next FB (overlapped with optim)
    fb_future = await client.forward_backward_async(batches[i], "ppo")
    
    # Submit and await optim step
    opt_future = await client.optim_step_async()
    opt_result = await opt_future
```

## Architecture Details

### Request Queue

Signal maintains an **ordered request queue** on the server:

- **FIFO ordering**: Requests processed in submission order
- **Concurrent execution**: Multiple requests can run simultaneously
- **Status tracking**: Each request has a unique ID and status

Request statuses:
- `queued`: Waiting in queue
- `running`: Currently executing
- `completed`: Finished successfully
- `failed`: Encountered an error

### Checking Request Status

```python
# Get status of a specific request
status = await client._check_future_status(request_id)

print(f"Status: {status['status']}")
if status['status'] == 'completed':
    result = status['result']
```

### Queue Statistics API

```python
# Via API endpoint
GET /runs/{run_id}/queue/stats

# Returns:
{
    "run_id": "run_123",
    "total": 10,
    "queued": 3,
    "running": 2,
    "completed": 5,
    "failed": 0
}
```

## Performance Benefits

### Latency Hiding

By submitting requests while others execute:
- **Network latency** overlaps with **GPU computation**
- **Data transfer** overlaps with **forward/backward pass**
- **Request processing** overlaps with **previous execution**

### Throughput Maximization

Without pipelining:
```
Request 1: [Submit] --wait--> [Execute] --wait--> [Result]
Request 2:                                         [Submit] --wait--> [Execute]
Total: 20 seconds
```

With pipelining:
```
Request 1: [Submit] ---> [Execute] ----> [Result]
Request 2:   [Submit] ---> [Execute] ---> [Result]
Request 3:     [Submit] ---> [Execute] --> [Result]
Total: 12 seconds (40% faster!)
```

### GPU Utilization

Pipelining ensures:
- No idle cycles between requests
- Continuous GPU workload
- Maximum cluster efficiency

## Concurrency Control

### Semaphore-Based Backpressure

Prevent overwhelming the server:

```python
import asyncio

max_concurrent = 3
semaphore = asyncio.Semaphore(max_concurrent)

async def controlled_request(batch):
    async with semaphore:
        future = await client.forward_backward_async(batch, "ppo")
        return await future

# Use in training loop
results = await asyncio.gather(*[
    controlled_request(batch)
    for batch in batches
])
```

### Client Configuration

```python
client = AsyncTrainingClientV2(
    run_id="run_123",
    api_key="your-api-key",
    max_concurrent_requests=3,  # Max in-flight requests
)
```

## Error Handling

### Future Cancellation

```python
try:
    future = await client.forward_backward_async(batch, "ppo")
    result = await future
except asyncio.CancelledError:
    # Handle cancellation
    future.cancel()
```

### Timeout Management

```python
# Per-operation timeout
future = await client.forward_backward_async(batch, "ppo")
try:
    result = await asyncio.wait_for(future, timeout=300)  # 5 minutes
except asyncio.TimeoutError:
    # Handle timeout
    future.cancel()
```

### Error Propagation

```python
futures = []
for batch in batches:
    future = await client.forward_backward_async(batch, "ppo")
    futures.append(future)

# Gather with exception handling
results = await asyncio.gather(*futures, return_exceptions=True)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Request {i} failed: {result}")
    else:
        print(f"Request {i} succeeded: loss={result['loss']}")
```

## Best Practices

1. **Submit early**: Submit next request while previous executes
2. **Batch submissions**: Submit multiple requests before awaiting
3. **Monitor queue depth**: Use `get_metrics()` to track pending requests
4. **Handle errors**: Use `return_exceptions=True` with `asyncio.gather()`
5. **Limit concurrency**: Use semaphore to prevent queue overflow
6. **Clean up futures**: Cancel futures if operation is aborted

## Comparison with Tinker

| Feature | Tinker | Signal |
|---------|--------|--------|
| **Async Pattern** | Double-await | Double-await |
| **Request Queue** | Server-side | Server-side |
| **Ordering** | Guaranteed | Guaranteed |
| **Feature Flag** | Always on | `SIGNAL_ENABLE_FUTURES=true` |
| **Infrastructure** | Managed cluster | Modal containers |
| **Execution Model** | Discrete cycles (~10s) | Stateful containers |

## Migration Guide

### From Single-Await (V1)

**Before (V1):**
```python
from rewardsignal import AsyncTrainingClient

client = AsyncTrainingClient(run_id, api_key)
result = await client.forward_backward(batch, "ppo")
```

**After (V2 with futures):**
```python
from rewardsignal.async_training_client_v2 import AsyncTrainingClientV2

client = AsyncTrainingClientV2(run_id, api_key, enable_futures=True)
future = await client.forward_backward_async(batch, "ppo")
result = await future
```

### Backward Compatibility

V2 client with `enable_futures=False` (default) behaves like V1:
```python
client = AsyncTrainingClientV2(run_id, api_key, enable_futures=False)
future = await client.forward_backward_async(batch, "ppo")
# Future is already completed, second await is instant
result = await future
```

## See Also

- [RL Algorithms Documentation](RL-Algorithms.md)
- [Metrics & Monitoring](Metrics.md)
- [Examples: Futures Pipelining](../client/examples/futures_pipelining.py)

