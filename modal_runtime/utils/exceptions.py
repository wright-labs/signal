"""Custom exception types for the Modal runtime."""
from typing import Dict, Optional


class ModalRuntimeException(Exception):
    """Base exception for errors that should surface through Modal."""

    def __init__(
        self,
        message: str = "Modal runtime error",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.payload: Dict[str, Any] = payload or {}


class OOMError(ModalRuntimeException):
    """Raised when training or inference hits an out-of-memory condition."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__("Out of memory error", payload=payload)
