import asyncio
import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

import psutil
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import start_http_server

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Metrics collector for monitoring system performance."""

    def __init__(self, port: int = 8000):
        """Initialize metrics collector.

        Args:
            port: Port for metrics server
        """
        # Request metrics
        self.request_count = Counter(
            "request_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        self.request_latency = Histogram(
            "request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        # Service metrics
        self.service_health = Gauge(
            "service_health", "Service health status", ["service"]
        )

        self.service_latency = Histogram(
            "service_latency_seconds",
            "Service latency in seconds",
            ["service", "operation"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        # System metrics
        self.cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")

        self.memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")

        self.disk_usage = Gauge("disk_usage_bytes", "Disk usage in bytes")

        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}

        # Start metrics server
        start_http_server(port)

        # Start system metrics collection
        asyncio.create_task(self._collect_system_metrics())

    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                # CPU usage
                self.cpu_usage.set(psutil.cpu_percent())

                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)

                # Disk usage
                disk = psutil.disk_usage("/")
                self.disk_usage.set(disk.used)

                await asyncio.sleep(60)  # Collect every minute

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)

    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status: Response status code
            duration: Request duration in seconds
        """
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)

    def track_service_health(self, service: str, is_healthy: bool):
        """Track service health status.

        Args:
            service: Service name
            is_healthy: Health status
        """
        self.service_health.labels(service=service).set(1 if is_healthy else 0)

    def track_service_latency(self, service: str, operation: str, duration: float):
        """Track service operation latency.

        Args:
            service: Service name
            operation: Operation name
            duration: Operation duration in seconds
        """
        self.service_latency.labels(service=service, operation=operation).observe(
            duration
        )

    def create_counter(
        self, name: str, description: str, labels: Optional[list] = None
    ):
        """Create a new counter metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        if name in self.custom_metrics:
            return self.custom_metrics[name]

        counter = Counter(name, description, labels or [])
        self.custom_metrics[name] = counter
        return counter

    def create_gauge(self, name: str, description: str, labels: Optional[list] = None):
        """Create a new gauge metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        if name in self.custom_metrics:
            return self.custom_metrics[name]

        gauge = Gauge(name, description, labels or [])
        self.custom_metrics[name] = gauge
        return gauge

    def create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[list] = None,
        buckets: Optional[tuple] = None,
    ):
        """Create a new histogram metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: Histogram buckets
        """
        if name in self.custom_metrics:
            return self.custom_metrics[name]

        histogram = Histogram(name, description, labels or [], buckets=buckets)
        self.custom_metrics[name] = histogram
        return histogram

    def create_summary(
        self, name: str, description: str, labels: Optional[list] = None
    ):
        """Create a new summary metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        if name in self.custom_metrics:
            return self.custom_metrics[name]

        summary = Summary(name, description, labels or [])
        self.custom_metrics[name] = summary
        return summary

    def track_request_decorator(self, method: str, endpoint: str):
        """Decorator for tracking request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint

        Returns:
            Decorated function
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    response = await func(*args, **kwargs)
                    status = getattr(response, "status_code", 200)
                    self.track_request(
                        method, endpoint, status, time.time() - start_time
                    )
                    return response
                except Exception:
                    self.track_request(method, endpoint, 500, time.time() - start_time)
                    raise

            return wrapper

        return decorator

    def track_service_decorator(self, service: str, operation: str):
        """Decorator for tracking service metrics.

        Args:
            service: Service name
            operation: Operation name

        Returns:
            Decorated function
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    self.track_service_latency(
                        service, operation, time.time() - start_time
                    )
                    return result
                except Exception:
                    self.track_service_latency(
                        service, operation, time.time() - start_time
                    )
                    raise

            return wrapper

        return decorator

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "requests": {
                "total": self.request_count._value.get(),
                "latency": self.request_latency._sum.get(),
            },
            "services": {
                "health": {k: v.get() for k, v in self.service_health._metrics.items()},
                "latency": {
                    k: v._sum.get() for k, v in self.service_latency._metrics.items()
                },
            },
            "system": {
                "cpu": self.cpu_usage._value.get(),
                "memory": self.memory_usage._value.get(),
                "disk": self.disk_usage._value.get(),
            },
            "custom": {
                name: metric._value.get() if hasattr(metric, "_value") else None
                for name, metric in self.custom_metrics.items()
            },
        }
        return metrics
