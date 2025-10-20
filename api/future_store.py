"""Future storage for async execution.

Stores Modal futures and provides lookup by ID.
In-memory storage for now, could be moved to Redis for production.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading


class FutureStore:
    """In-memory future storage with TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize future store. Time-to-live for futures (default 1 hour)"""
        self._futures: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_seconds

    def store(
        self, future_id: str, modal_future: Any, metadata: Optional[Dict] = None
    ) -> None:
        """Store a Modal future."""
        with self._lock:
            self._futures[future_id] = {
                "future": modal_future,
                "created_at": datetime.utcnow(),
                "metadata": metadata or {},
            }

    def get(self, future_id: str) -> Any:
        """Get a Modal future by ID."""
        with self._lock:
            if future_id not in self._futures:
                raise KeyError(f"Future {future_id} not found")

            # Check TTL
            entry = self._futures[future_id]
            age = datetime.utcnow() - entry["created_at"]
            if age > timedelta(seconds=self.ttl_seconds):
                del self._futures[future_id]
                raise KeyError(f"Future {future_id} expired")

            return entry["future"]

    def delete(self, future_id: str) -> bool:
        """Delete a future."""
        with self._lock:
            if future_id in self._futures:
                del self._futures[future_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove expired futures."""
        with self._lock:
            now = datetime.utcnow()
            expired = [
                fid
                for fid, entry in self._futures.items()
                if now - entry["created_at"] > timedelta(seconds=self.ttl_seconds)
            ]
            for fid in expired:
                del self._futures[fid]
            return len(expired)

    def count(self) -> int:
        """Get the number of stored futures."""
        with self._lock:
            return len(self._futures)


_future_store = FutureStore()


def store_future(
    future_id: str, modal_future: Any, metadata: Optional[Dict] = None
) -> None:
    """Store a Modal future."""
    _future_store.store(future_id, modal_future, metadata)


def get_future(future_id: str) -> Any:
    """Get a Modal future by ID."""
    return _future_store.get(future_id)


def delete_future(future_id: str) -> bool:
    """Delete a future."""
    return _future_store.delete(future_id)


def cleanup_expired_futures() -> int:
    """Cleanup expired futures."""
    return _future_store.cleanup_expired()
