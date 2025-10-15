"""Datadog metrics client for Signal training metrics.

This module provides a centralized client for sending metrics to Datadog.
Supports gauge, histogram, counter, and increment metrics.
"""

import os
from typing import List, Optional, Dict, Any
import time


class DatadogMetrics:
    """Datadog metrics client.
    
    This client uses the Datadog statsd client to send metrics.
    Automatically adds common tags and handles initialization.
    
    Example:
        metrics = DatadogMetrics()
        metrics.gauge("signal.training.loss", 0.5, tags=["run_id:123"])
        metrics.histogram("signal.performance.forward_backward_duration_ms", 250.5)
    """
    
    def __init__(
        self,
        enabled: Optional[bool] = None,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 8125,
        namespace: str = "signal",
        constant_tags: Optional[List[str]] = None,
    ):
        """Initialize Datadog metrics client.
        
        Args:
            enabled: Whether to enable Datadog (default: read from env DATADOG_ENABLED)
            api_key: Datadog API key (default: read from env DATADOG_API_KEY)
            app_key: Datadog app key (default: read from env DATADOG_APP_KEY)
            host: Datadog agent host (default: localhost)
            port: Datadog agent port (default: 8125)
            namespace: Metric namespace prefix (default: "signal")
            constant_tags: Tags to add to all metrics
        """
        # Determine if Datadog is enabled
        if enabled is None:
            enabled = os.getenv("DATADOG_ENABLED", "false").lower() == "true"
        self.enabled = enabled
        
        # Get API keys from environment
        if api_key is None:
            api_key = os.getenv("DATADOG_API_KEY")
        if app_key is None:
            app_key = os.getenv("DATADOG_APP_KEY")
        
        self.api_key = api_key
        self.app_key = app_key
        self.host = host
        self.port = port
        self.namespace = namespace
        self.constant_tags = constant_tags or []
        
        # Statsd client
        self.statsd = None
        
        # Initialize if enabled
        if self.enabled:
            self._initialize()
        else:
            print("Datadog metrics disabled (set DATADOG_ENABLED=true to enable)")
    
    def _initialize(self) -> None:
        """Initialize Datadog client."""
        try:
            from datadog import initialize, statsd
            
            # Initialize Datadog
            initialize(
                api_key=self.api_key,
                app_key=self.app_key,
                statsd_host=self.host,
                statsd_port=self.port,
            )
            
            self.statsd = statsd
            
            # Configure namespace and constant tags
            if self.namespace:
                self.statsd.namespace = self.namespace
            if self.constant_tags:
                self.statsd.constant_tags = self.constant_tags
            
            print(f"✓ Datadog metrics initialized (namespace={self.namespace})")
            
        except ImportError:
            print("⚠️ Datadog library not installed. Run: pip install datadog")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ Failed to initialize Datadog: {e}")
            self.enabled = False
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Send a gauge metric.
        
        Gauges represent a value at a point in time.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Metric value
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.gauge(
                metric,
                value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending gauge metric {metric}: {e}")
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Send a histogram metric.
        
        Histograms track the statistical distribution of values.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Metric value
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.histogram(
                metric,
                value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending histogram metric {metric}: {e}")
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Increment a counter metric.
        
        Counters track the total number of events.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Increment amount (default: 1)
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.increment(
                metric,
                value=value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending increment metric {metric}: {e}")
    
    def decrement(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Decrement a counter metric.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Decrement amount (default: 1)
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.decrement(
                metric,
                value=value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending decrement metric {metric}: {e}")
    
    def timing(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Send a timing metric (in milliseconds).
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Time in milliseconds
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.timing(
                metric,
                value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending timing metric {metric}: {e}")
    
    def distribution(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Send a distribution metric.
        
        Distributions provide global percentile aggregations.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Metric value
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.distribution(
                metric,
                value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending distribution metric {metric}: {e}")
    
    def set_metric(
        self,
        metric: str,
        value: Any,
        tags: Optional[List[str]] = None,
        sample_rate: float = 1.0,
    ) -> None:
        """Send a set metric.
        
        Sets count the number of unique values.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            value: Unique value to count
            tags: Optional tags
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.set(
                metric,
                value,
                tags=tags,
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Error sending set metric {metric}: {e}")
    
    def timed(self, metric: str, tags: Optional[List[str]] = None):
        """Decorator to time a function and send as a metric.
        
        Args:
            metric: Metric name (will be prefixed with namespace)
            tags: Optional tags
            
        Returns:
            Decorator function
        
        Example:
            @metrics.timed("my_function_duration", tags=["env:prod"])
            def my_function():
                pass
        """
        if not self.enabled or self.statsd is None:
            # Return no-op decorator if disabled
            def decorator(func):
                return func
            return decorator
        
        return self.statsd.timed(metric, tags=tags)
    
    def event(
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
            tags: Optional tags
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            self.statsd.event(
                title,
                text,
                alert_type=alert_type,
                tags=tags,
            )
        except Exception as e:
            print(f"Error sending event {title}: {e}")
    
    def service_check(
        self,
        check_name: str,
        status: str,
        tags: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        """Send a service check to Datadog.
        
        Args:
            check_name: Check name
            status: Check status ("OK", "WARNING", "CRITICAL", "UNKNOWN")
            tags: Optional tags
            message: Optional message
        """
        if not self.enabled or self.statsd is None:
            return
        
        try:
            # Convert status string to integer code
            status_map = {
                "OK": 0,
                "WARNING": 1,
                "CRITICAL": 2,
                "UNKNOWN": 3,
            }
            status_code = status_map.get(status, 3)
            
            self.statsd.service_check(
                check_name,
                status_code,
                tags=tags,
                message=message,
            )
        except Exception as e:
            print(f"Error sending service check {check_name}: {e}")


# Global metrics instance (singleton)
_global_metrics: Optional[DatadogMetrics] = None


def get_metrics_client() -> DatadogMetrics:
    """Get the global Datadog metrics client.
    
    Returns:
        Global DatadogMetrics instance
    """
    global _global_metrics
    
    if _global_metrics is None:
        _global_metrics = DatadogMetrics()
    
    return _global_metrics

