"""Frontier Signal SDK - Python client for the Frontier Signal training API."""

from ._version import __version__

# Sync client
from .client import SignalClient, SignalRun

# Specialized clients
from .training_client import TrainingClient
from .inference_client import InferenceClient

# Async client
from .async_client import AsyncSignalClient, AsyncSignalRun

# Futures
from .futures import SignalFuture, FutureGroup

# Exceptions
from .exceptions import (
    SignalAPIError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    ConnectionError,
    TimeoutError,
)

# Schemas (for type hints)
from .schemas import (
    RunConfig,
    RunResponse,
    RunStatus,
    RunMetrics,
    TrainingExample,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    OptimStepRequest,
    OptimStepResponse,
    SampleRequest,
    SampleResponse,
    SaveStateRequest,
    SaveStateResponse,
    ErrorResponse,
)

__all__ = [
    # Version
    "__version__",
    # Sync client
    "SignalClient",
    "SignalRun",
    # Specialized clients
    "TrainingClient",
    "InferenceClient",
    # Async client
    "AsyncSignalClient",
    "AsyncSignalRun",
    # Futures
    "SignalFuture",
    "FutureGroup",
    # Exceptions
    "SignalAPIError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "ConnectionError",
    "TimeoutError",
    # Schemas
    "RunConfig",
    "RunResponse",
    "RunStatus",
    "RunMetrics",
    "TrainingExample",
    "ForwardBackwardRequest",
    "ForwardBackwardResponse",
    "OptimStepRequest",
    "OptimStepResponse",
    "SampleRequest",
    "SampleResponse",
    "SaveStateRequest",
    "SaveStateResponse",
    "ErrorResponse",
]
