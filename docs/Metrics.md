# Metrics & Monitoring

Signal provides comprehensive metrics collection and Datadog integration for monitoring training, RL algorithms, and system performance.

## Overview

Signal tracks three categories of metrics:

1. **Training Metrics**: Loss, gradients, learning rate, steps
2. **RL Metrics**: Policy loss, value loss, KL divergence, entropy, advantages
3. **Performance Metrics**: Latency, GPU utilization, queue depth

All metrics can be sent to **Datadog** for visualization and alerting.

## Datadog Integration

### Setup

1. **Install Datadog library**:
```bash
pip install datadog
```

2. **Set environment variables**:
```bash
export DATADOG_ENABLED=true
export DATADOG_API_KEY="your-datadog-api-key"
export DATADOG_APP_KEY="your-datadog-app-key"
```

3. **Metrics automatically sent**:
Once enabled, Signal automatically sends metrics to Datadog during training.

### Configuration

```python
from api.datadog_client import DatadogMetrics

metrics = DatadogMetrics(
    enabled=True,
    api_key="your-api-key",
    app_key="your-app-key",
    namespace="signal",  # Metric prefix
    constant_tags=["env:production", "team:ml"],
)
```

## Training Metrics

### Core Training Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `signal.training.loss` | Gauge | Training loss | float |
| `signal.training.grad_norm` | Gauge | Gradient L2 norm | float |
| `signal.training.learning_rate` | Gauge | Current learning rate | float |
| `signal.training.step` | Counter | Training step count | int |

### Usage

```python
from modal_runtime.metrics import create_metrics_collector

collector = create_metrics_collector(
    run_id="run_123",
    user_id="user_456",
    enable_datadog=True,
)

# Collect training metrics
collector.collect_training_metrics(
    loss=0.5,
    grad_norm=2.3,
    learning_rate=1e-4,
    step=100,
    extra_metrics={
        "perplexity": 5.2,
        "tokens_per_second": 1500,
    },
)
```

### Visualization

**Datadog Dashboard**:
```
Training Loss over Time
┌─────────────────────────────┐
│     ╱╲                      │
│    ╱  ╲  ╱╲                 │
│   ╱    ╲╱  ╲╱╲              │
│  ╱           ╲              │
│ ╱             ╲──           │
└─────────────────────────────┘
  0        50       100   steps
```

## RL Metrics

### Available RL Metrics

| Metric | Type | Description | Typical Range |
|--------|------|-------------|---------------|
| `signal.rl.policy_loss` | Gauge | PPO policy loss | -10 to 10 |
| `signal.rl.value_loss` | Gauge | Value function MSE | 0 to 100 |
| `signal.rl.entropy` | Gauge | Policy entropy | 0 to 10 |
| `signal.rl.kl_divergence` | Gauge | KL from reference | 0 to 0.1 |
| `signal.rl.clip_fraction` | Gauge | PPO clip activation | 0 to 1 |
| `signal.rl.advantage_mean` | Gauge | Mean advantage | -5 to 5 |
| `signal.rl.advantage_std` | Gauge | Advantage std dev | 0 to 5 |
| `signal.rl.explained_variance` | Gauge | Value function quality | -1 to 1 |
| `signal.rl.reward_mean` | Gauge | Mean reward | varies |

### Usage

```python
collector.collect_rl_metrics(
    step=100,
    policy_loss=-2.3,
    value_loss=0.5,
    entropy=3.2,
    kl_divergence=0.01,
    clip_fraction=0.2,
    advantage_mean=0.1,
    advantage_std=0.9,
    explained_variance=0.8,
    reward_mean=5.2,
)
```

### Monitoring RL Training

**Key metrics to watch**:

1. **KL Divergence**:
   - Normal: < 0.05
   - Warning: 0.05 - 0.1
   - Critical: > 0.1

2. **Clip Fraction**:
   - Optimal: 0.1 - 0.3
   - Too low (<0.1): Reduce `clip_epsilon`
   - Too high (>0.5): Increase `clip_epsilon` or reduce LR

3. **Explained Variance**:
   - Good: > 0.7
   - Acceptable: 0.5 - 0.7
   - Poor: < 0.5 (value function not learning)

4. **Entropy**:
   - Decreasing over time is normal
   - Sudden drops indicate policy collapse
   - Use entropy bonus to maintain exploration

### Datadog Alerts

```python
# Alert if KL divergence too high
if kl_divergence > 0.1:
    metrics.event(
        title="High KL Divergence Detected",
        text=f"KL divergence {kl_divergence:.4f} exceeds threshold 0.1",
        alert_type="warning",
        tags=[f"run_id:{run_id}"],
    )
```

## Performance Metrics

### Available Performance Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `signal.performance.forward_backward_duration_ms` | Histogram | FB pass duration | milliseconds |
| `signal.performance.optim_step_duration_ms` | Histogram | Optim step duration | milliseconds |
| `signal.performance.queue_depth` | Gauge | Queued requests | int |
| `signal.performance.gpu_utilization` | Gauge | GPU usage | 0-100% |
| `signal.performance.gpu_memory_used_gb` | Gauge | GPU memory used | GB |

### Usage

```python
collector.collect_performance_metrics(
    forward_backward_duration_ms=250.5,
    optim_step_duration_ms=50.2,
    queue_depth=3,
    gpu_utilization=95.5,
    gpu_memory_used_gb=38.2,
)
```

### Timing Operations

```python
# Time an operation
with collector.time_operation("forward_backward"):
    # ... do forward backward ...
    result = await training_client.forward_backward(batch, "ppo")

# Automatically sends metric:
# signal.performance.forward_backward_duration_ms
```

## Metrics in Training Loop

### Example Integration

```python
import asyncio
from rewardsignal import AsyncSignalClient
from modal_runtime.metrics import create_metrics_collector

async def train_with_metrics():
    client = AsyncSignalClient(api_key="your-api-key")
    run = await client.create_run(base_model="meta-llama/Llama-3.2-1B")
    
    # Create metrics collector
    collector = create_metrics_collector(
        run_id=run["run_id"],
        user_id="user_123",
        enable_datadog=True,
    )
    
    training_client = client.get_training_client(run["run_id"])
    
    for step in range(100):
        # Time forward-backward
        with collector.time_operation("forward_backward"):
            result = await training_client.forward_backward(
                batch_data=batch,
                loss_fn="enhanced_ppo",
            )
        
        # Collect training metrics
        collector.collect_training_metrics(
            loss=result["loss"],
            grad_norm=result["grad_norm"],
            learning_rate=1e-5,
            step=step,
        )
        
        # Collect RL metrics
        if "metrics" in result:
            metrics = result["metrics"]
            collector.collect_rl_metrics(
                step=step,
                policy_loss=metrics.get("policy_loss"),
                value_loss=metrics.get("value_loss"),
                entropy=metrics.get("entropy"),
                kl_divergence=metrics.get("kl_divergence"),
                clip_fraction=metrics.get("clip_fraction"),
            )
        
        # Time optimizer step
        with collector.time_operation("optim_step"):
            await training_client.optim_step()
        
        # Log every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: loss={result['loss']:.4f}")
```

## Custom Metrics

### Adding Custom Metrics

```python
# Send custom gauge
metrics.gauge("signal.custom.my_metric", 42.0, tags=["run_id:123"])

# Send custom histogram
metrics.histogram("signal.custom.latency_ms", 125.3)

# Increment counter
metrics.increment("signal.custom.event_count", value=1)
```

### Custom Events

```python
# Send event to Datadog
metrics.event(
    title="Training Milestone Reached",
    text=f"Model reached 1000 steps with loss {loss:.4f}",
    alert_type="success",
    tags=[f"run_id:{run_id}", "milestone:1000"],
)
```

## Datadog Dashboards

### Pre-built Dashboard Query Examples

**Training Loss**:
```
avg:signal.training.loss{run_id:*}
```

**KL Divergence (alert threshold)**:
```
avg:signal.rl.kl_divergence{run_id:*}
```

**GPU Utilization**:
```
avg:signal.performance.gpu_utilization{*}
```

**Request Queue Depth**:
```
avg:signal.performance.queue_depth{*}
```

### Dashboard Layout

```
┌─────────────────────────────────────────────┐
│  Training Loss          │  Learning Rate    │
├─────────────────────────┼───────────────────┤
│  KL Divergence          │  Clip Fraction    │
├─────────────────────────┼───────────────────┤
│  GPU Utilization        │  Queue Depth      │
├─────────────────────────┴───────────────────┤
│  Forward-Backward Latency (p50, p99)        │
└─────────────────────────────────────────────┘
```

## Local Metrics (Non-Datadog)

If Datadog is not enabled, metrics are still collected locally:

```python
collector = create_metrics_collector(
    run_id="run_123",
    enable_datadog=False,  # Disable Datadog
)

# Metrics stored in memory
collector.collect_training_metrics(...)

# Get metrics summary
summary = collector.get_metrics_summary()
print(f"Steps: {summary['last_step']}")
print(f"Loss: {summary['last_loss']}")
print(f"Total metrics: {summary['num_metrics']}")
```

## Best Practices

### 1. Use Tags for Filtering

```python
tags = [
    f"run_id:{run_id}",
    f"user_id:{user_id}",
    f"model:{base_model}",
    "env:production",
]

metrics.gauge("signal.training.loss", loss, tags=tags)
```

### 2. Sample Rates for High-Volume Metrics

```python
# Sample only 10% of requests
metrics.histogram(
    "signal.performance.forward_backward_duration_ms",
    duration_ms,
    sample_rate=0.1,  # 10% sampling
)
```

### 3. Alert on Critical Metrics

```python
# Monitor KL divergence
if kl_divergence > 0.1:
    metrics.service_check(
        check_name="rl.kl_divergence",
        status="CRITICAL",
        message=f"KL divergence {kl_divergence:.4f} exceeds 0.1",
        tags=[f"run_id:{run_id}"],
    )
```

### 4. Use Distributions for Percentiles

```python
# Use distribution for p50, p95, p99
metrics.distribution(
    "signal.performance.forward_backward_duration_ms",
    duration_ms,
)
```

## Troubleshooting

### Datadog Not Receiving Metrics

**Check**:
1. Environment variables set correctly
2. Datadog agent running (if using StatsD)
3. API keys valid
4. Network connectivity

```python
# Test connection
metrics = DatadogMetrics()
metrics.gauge("test.metric", 1.0)
# Check Datadog UI for "test.metric"
```

### Metrics Not Appearing in Dashboard

**Check**:
1. Namespace prefix correct (`signal.*`)
2. Tags filtering correctly
3. Time range in dashboard
4. Metrics aggregation settings

### High Memory Usage

**Solution**: Clear metrics history periodically

```python
collector.metrics_history.clear()
```

## API Reference

### MetricsCollector

```python
class MetricsCollector:
    def __init__(
        self,
        run_id: str,
        user_id: Optional[str] = None,
        enable_datadog: bool = True,
    )
    
    def collect_training_metrics(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        step: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    )
    
    def collect_rl_metrics(
        self,
        step: int,
        policy_loss: Optional[float] = None,
        # ... other RL metrics
    )
    
    def collect_performance_metrics(
        self,
        forward_backward_duration_ms: Optional[float] = None,
        # ... other performance metrics
    )
    
    def time_operation(
        self,
        operation_name: str,
    ) -> ContextManager
```

### DatadogMetrics

```python
class DatadogMetrics:
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None,
    )
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None,
    )
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[List[str]] = None,
    )
    
    def event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[List[str]] = None,
    )
```

## See Also

- [RL Algorithms](RL-Algorithms.md)
- [Futures Architecture](Futures-Architecture.md)
- [Reference Models](Reference-Models.md)

