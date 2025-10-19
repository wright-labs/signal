"""Training metrics collection and reporting.

This module provides utilities for collecting and reporting training metrics
to Datadog and other monitoring systems.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class MetricsCollector:
    """Collects training metrics and sends to monitoring systems."""
    
    def __init__(
        self,
        run_id: str,
        user_id: Optional[str] = None,
        enable_datadog: bool = True,
    ):
        """Initialize metrics collector."""
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
        """Get tags for metrics."""
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
        """Collect basic training metrics."""
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
        """Collect RL-specific metrics. """
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
        """Collect performance metrics."""
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
        """Context manager to time an operation."""
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
        """Send an event to Datadog."""
        if self.datadog:
            metric_tags = self._get_tags(tags)
            self.datadog.event(title, text, alert_type=alert_type, tags=metric_tags)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
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
    """Create a metrics collector instance."""
    return MetricsCollector(
        run_id=run_id,
        user_id=user_id,
        enable_datadog=enable_datadog,
    )

