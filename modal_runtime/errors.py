"""Custom exceptions raised inside Modal runtime."""

from __future__ import annotations

from typing import Any, Dict, Optional


class OOMError(RuntimeError):
    """Exception raised when a training step runs out of GPU memory."""

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.payload: Dict[str, Any] = payload or {}

    def __reduce__(self):
        # Ensure the payload survives pickling/deserialization across Modal transport
        return (self.__class__, (self.args[0], self.payload))

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the error payload."""

        data = {"message": self.args[0]}
        if self.payload:
            data["payload"] = self.payload
        return data
