"""Training metrics collection for WandB and monitoring.

This module provides simple metrics collection for training runs.
Uses WandB for experiment tracking (already installed in app.py).
"""
import time
from typing import Dict, Any, Optional, List


class MetricsCollector:
    """Simple metrics collector for training runs."""
    
    def __init__(
        self,
        run_id: str,
        user_id: Optional[str] = None,
        enable_wandb: bool = True,
    ):
        """Initialize metrics collector."""
        self.run_id = run_id
        self.user_id = user_id
        self.enable_wandb = enable_wandb
        
        # WandB client (lazy-loaded)
        self._wandb = None
        
        # Local metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
    
    @property
    def wandb(self):
        """Get WandB client (lazy-loaded)."""
        if self._wandb is None and self.enable_wandb:
            try:
                import wandb
                self._wandb = wandb
            except Exception as e:
                print(f"Failed to import WandB: {e}")
                self.enable_wandb = False
        
        return self._wandb
    
    def collect_training_metrics(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        step: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Collect basic training metrics."""
        metrics_dict = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "timestamp": time.time(),
        }
        
        if extra_metrics:
            metrics_dict.update(extra_metrics)
        
        # Log to WandB
        if self.wandb and self.enable_wandb:
            self.wandb.log(metrics_dict)
        
        # Store locally
        self.metrics_history.append(metrics_dict)
    
    def collect_rl_metrics(
        self,
        step: int,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        reward_mean: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Collect RL-specific metrics."""
        metrics_dict = {
            "step": step,
            "timestamp": time.time(),
        }
        
        if policy_loss is not None:
            metrics_dict["rl/policy_loss"] = policy_loss
        if value_loss is not None:
            metrics_dict["rl/value_loss"] = value_loss
        if entropy is not None:
            metrics_dict["rl/entropy"] = entropy
        if kl_divergence is not None:
            metrics_dict["rl/kl_divergence"] = kl_divergence
        if reward_mean is not None:
            metrics_dict["rl/reward_mean"] = reward_mean
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                metrics_dict[f"rl/{key}"] = value
        
        # Log to WandB
        if self.wandb and self.enable_wandb:
            self.wandb.log(metrics_dict)
        
        # Store locally
        self.metrics_history.append(metrics_dict)
    
    def collect_performance_metrics(
        self,
        forward_backward_duration_ms: Optional[float] = None,
        optim_step_duration_ms: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        gpu_memory_used_gb: Optional[float] = None,
    ) -> None:
        """Collect performance metrics."""
        metrics_dict = {
            "timestamp": time.time(),
        }
        
        if forward_backward_duration_ms is not None:
            metrics_dict["perf/forward_backward_ms"] = forward_backward_duration_ms
        if optim_step_duration_ms is not None:
            metrics_dict["perf/optim_step_ms"] = optim_step_duration_ms
        if gpu_utilization is not None:
            metrics_dict["perf/gpu_utilization"] = gpu_utilization
        if gpu_memory_used_gb is not None:
            metrics_dict["perf/gpu_memory_gb"] = gpu_memory_used_gb
        
        # Log to WandB
        if self.wandb and self.enable_wandb:
            self.wandb.log(metrics_dict)
    
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
    enable_wandb: bool = True,
) -> MetricsCollector:
    """Create a metrics collector instance."""
    return MetricsCollector(
        run_id=run_id,
        user_id=user_id,
        enable_wandb=enable_wandb,
    )
