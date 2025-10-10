"""Frontier Signal SDK - Python client for the Frontier Signal training API."""

from ._version import __version__

# Sync client
from .client import SignalClient, SignalRun

# Async client
from .async_client import AsyncSignalClient, AsyncSignalRun

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
    # Async client
    "AsyncSignalClient",
    "AsyncSignalRun",
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
