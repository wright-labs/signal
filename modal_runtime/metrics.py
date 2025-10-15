"""Training metrics collection and reporting.

This module provides utilities for collecting and reporting training metrics
to Datadog and other monitoring systems.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class MetricsCollector:
    """Collects training metrics and sends to monitoring systems.
    
    This class:
    - Collects metrics from training loops
    - Sends metrics to Datadog (if enabled)
    - Provides timing context managers
    - Aggregates metrics over time
    
    Example:
        collector = MetricsCollector(run_id="run_123")
        
        # Collect training metrics
        collector.collect_training_metrics(
            loss=0.5,
            grad_norm=2.3,
            learning_rate=1e-4,
            step=100,
        )
        
        # Time an operation
        with collector.time_operation("forward_backward"):
            # ... do forward backward ...
            pass
    """
    
    def __init__(
        self,
        run_id: str,
        user_id: Optional[str] = None,
        enable_datadog: bool = True,
    ):
        """Initialize metrics collector.
        
        Args:
            run_id: Run identifier
            user_id: User identifier (optional)
            enable_datadog: Whether to send metrics to Datadog
        """
        self.run_id = run_id
        self.user_id = user_id
        self.enable_datadog = enable_datadog
        
        # Datadog client (lazy-loaded)
        self._datadog_client = None
        
        # Local metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
    
    @property
    def datadog(self):
        """Get Datadog client (lazy-loaded)."""
        if self._datadog_client is None and self.enable_datadog:
            try:
                # Import here to avoid circular dependency
                import sys
                import os
                
                # Add api directory to path if needed
                api_path = os.path.join(os.path.dirname(__file__), "..", "api")
                if api_path not in sys.path:
                    sys.path.insert(0, api_path)
                
                from datadog_client import get_metrics_client
                self._datadog_client = get_metrics_client()
            except Exception as e:
                print(f"Failed to load Datadog client: {e}")
                self.enable_datadog = False
        
        return self._datadog_client
    
    def _get_tags(self, extra_tags: Optional[List[str]] = None) -> List[str]:
        """Get tags for metrics.
        
        Args:
            extra_tags: Additional tags
            
        Returns:
            List of tags
        """
        tags = [f"run_id:{self.run_id}"]
        
        if self.user_id:
            tags.append(f"user_id:{self.user_id}")
        
        if extra_tags:
            tags.extend(extra_tags)
        
        return tags
    
    def collect_training_metrics(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        step: int,
        extra_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Collect basic training metrics.
        
        Args:
            loss: Training loss
            grad_norm: Gradient norm
            learning_rate: Learning rate
            step: Training step
            extra_metrics: Additional metrics
            tags: Additional tags
        """
        # Send to Datadog
        if self.datadog:
            metric_tags = self._get_tags(tags)
            
            self.datadog.gauge("training.loss", loss, tags=metric_tags)
            self.datadog.gauge("training.grad_norm", grad_norm, tags=metric_tags)
            self.datadog.gauge("training.learning_rate", learning_rate, tags=metric_tags)
            self.datadog.increment("training.step", value=1, tags=metric_tags)
            
            # Send extra metrics
            if extra_metrics:
                for key, value in extra_metrics.items():
                    self.datadog.gauge(f"training.{key}", value, tags=metric_tags)
        
        # Store locally
        metrics_dict = {
            "timestamp": time.time(),
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }
        if extra_metrics:
            metrics_dict.update(extra_metrics)
        
        self.metrics_history.append(metrics_dict)
    
    def collect_rl_metrics(
        self,
        step: int,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        advantage_mean: Optional[float] = None,
        advantage_std: Optional[float] = None,
        explained_variance: Optional[float] = None,
        reward_mean: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Collect RL-specific metrics.
        
        Args:
            step: Training step
            policy_loss: Policy loss
            value_loss: Value function loss
            entropy: Policy entropy
            kl_divergence: KL divergence from reference
            clip_fraction: Fraction of clipped ratios
            advantage_mean: Mean advantage
            advantage_std: Advantage std deviation
            explained_variance: Value function explained variance
            reward_mean: Mean reward
            extra_metrics: Additional metrics
            tags: Additional tags
        """
        if self.datadog:
            metric_tags = self._get_tags(tags)
            
            if policy_loss is not None:
                self.datadog.gauge("rl.policy_loss", policy_loss, tags=metric_tags)
            if value_loss is not None:
                self.datadog.gauge("rl.value_loss", value_loss, tags=metric_tags)
            if entropy is not None:
                self.datadog.gauge("rl.entropy", entropy, tags=metric_tags)
            if kl_divergence is not None:
                self.datadog.gauge("rl.kl_divergence", kl_divergence, tags=metric_tags)
            if clip_fraction is not None:
                self.datadog.gauge("rl.clip_fraction", clip_fraction, tags=metric_tags)
            if advantage_mean is not None:
                self.datadog.gauge("rl.advantage_mean", advantage_mean, tags=metric_tags)
            if advantage_std is not None:
                self.datadog.gauge("rl.advantage_std", advantage_std, tags=metric_tags)
            if explained_variance is not None:
                self.datadog.gauge("rl.explained_variance", explained_variance, tags=metric_tags)
            if reward_mean is not None:
                self.datadog.gauge("rl.reward_mean", reward_mean, tags=metric_tags)
            
            # Send extra metrics
            if extra_metrics:
                for key, value in extra_metrics.items():
                    self.datadog.gauge(f"rl.{key}", value, tags=metric_tags)
    
    def collect_performance_metrics(
        self,
        forward_backward_duration_ms: Optional[float] = None,
        optim_step_duration_ms: Optional[float] = None,
        queue_depth: Optional[int] = None,
        gpu_utilization: Optional[float] = None,
        gpu_memory_used_gb: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Collect performance metrics.
        
        Args:
            forward_backward_duration_ms: Forward-backward duration in ms
            optim_step_duration_ms: Optimizer step duration in ms
            queue_depth: Number of queued requests
            gpu_utilization: GPU utilization percentage
            gpu_memory_used_gb: GPU memory used in GB
            tags: Additional tags
        """
        if self.datadog:
            metric_tags = self._get_tags(tags)
            
            if forward_backward_duration_ms is not None:
                self.datadog.histogram(
                    "performance.forward_backward_duration_ms",
                    forward_backward_duration_ms,
                    tags=metric_tags
                )
            if optim_step_duration_ms is not None:
                self.datadog.histogram(
                    "performance.optim_step_duration_ms",
                    optim_step_duration_ms,
                    tags=metric_tags
                )
            if queue_depth is not None:
                self.datadog.gauge("performance.queue_depth", queue_depth, tags=metric_tags)
            if gpu_utilization is not None:
                self.datadog.gauge("performance.gpu_utilization", gpu_utilization, tags=metric_tags)
            if gpu_memory_used_gb is not None:
                self.datadog.gauge("performance.gpu_memory_used_gb", gpu_memory_used_gb, tags=metric_tags)
    
    @contextmanager
    def time_operation(self, operation_name: str, tags: Optional[List[str]] = None):
        """Context manager to time an operation.
        
        Args:
            operation_name: Name of the operation
            tags: Additional tags
            
        Yields:
            None
            
        Example:
            with collector.time_operation("forward_backward"):
                # ... do forward backward ...
                pass
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            if self.datadog:
                metric_tags = self._get_tags(tags)
                self.datadog.histogram(
                    f"performance.{operation_name}_duration_ms",
                    duration_ms,
                    tags=metric_tags
                )
    
    def send_event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Send an event to Datadog.
        
        Args:
            title: Event title
            text: Event description
            alert_type: Alert type ("info", "warning", "error", "success")
            tags: Additional tags
        """
        if self.datadog:
            metric_tags = self._get_tags(tags)
            self.datadog.event(title, text, alert_type=alert_type, tags=metric_tags)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        if not self.metrics_history:
            return {
                "num_metrics": 0,
                "first_timestamp": None,
                "last_timestamp": None,
            }
        
        return {
            "num_metrics": len(self.metrics_history),
            "first_timestamp": self.metrics_history[0]["timestamp"],
            "last_timestamp": self.metrics_history[-1]["timestamp"],
            "last_step": self.metrics_history[-1].get("step"),
            "last_loss": self.metrics_history[-1].get("loss"),
        }


def create_metrics_collector(
    run_id: str,
    user_id: Optional[str] = None,
    enable_datadog: bool = True,
) -> MetricsCollector:
    """Create a metrics collector instance.
    
    Args:
        run_id: Run identifier
        user_id: User identifier (optional)
        enable_datadog: Whether to send metrics to Datadog
        
    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(
        run_id=run_id,
        user_id=user_id,
        enable_datadog=enable_datadog,
    )

