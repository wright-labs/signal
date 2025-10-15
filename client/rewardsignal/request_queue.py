"""Request queue management for client-side request tracking."""

import uuid
import time
from typing import Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field
import threading


@dataclass
class QueuedRequest:
    """Represents a queued request awaiting processing."""
    request_id: str
    request_type: str  # "forward_backward", "optim_step", "sample", etc.
    payload: Dict[str, Any]
    submitted_at: float = field(default_factory=time.time)
    status: str = "queued"  # "queued", "submitted", "running", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ClientRequestQueue:
    """Client-side request queue for tracking in-flight requests.
    
    This queue maintains ordering guarantees and tracks request status.
    Thread-safe for concurrent request submission.
    """
    
    def __init__(self, max_queue_size: int = 100):
        """Initialize request queue.
        
        Args:
            max_queue_size: Maximum number of queued requests
        """
        self.max_queue_size = max_queue_size
        self._queue: OrderedDict[str, QueuedRequest] = OrderedDict()
        self._lock = threading.Lock()
    
    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"req_{uuid.uuid4().hex[:16]}"
    
    def submit(
        self,
        request_type: str,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> str:
        """Submit a new request to the queue.
        
        Args:
            request_type: Type of request (e.g., "forward_backward")
            payload: Request payload
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            Request ID
            
        Raises:
            RuntimeError: If queue is full
        """
        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise RuntimeError(
                    f"Request queue full ({self.max_queue_size} requests). "
                    "Wait for requests to complete before submitting more."
                )
            
            if request_id is None:
                request_id = self.generate_request_id()
            
            request = QueuedRequest(
                request_id=request_id,
                request_type=request_type,
                payload=payload,
                status="queued",
            )
            
            self._queue[request_id] = request
            return request_id
    
    def update_status(
        self,
        request_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the status of a request.
        
        Args:
            request_id: Request ID
            status: New status
            result: Optional result data
            error: Optional error message
        """
        with self._lock:
            if request_id not in self._queue:
                return
            
            request = self._queue[request_id]
            request.status = status
            
            if result is not None:
                request.result = result
            
            if error is not None:
                request.error = error
    
    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Status dictionary or None if not found
        """
        with self._lock:
            if request_id not in self._queue:
                return None
            
            request = self._queue[request_id]
            return {
                "request_id": request.request_id,
                "request_type": request.request_type,
                "status": request.status,
                "result": request.result,
                "error": request.error,
                "submitted_at": request.submitted_at,
            }
    
    def remove(self, request_id: str) -> None:
        """Remove a request from the queue.
        
        Args:
            request_id: Request ID
        """
        with self._lock:
            if request_id in self._queue:
                del self._queue[request_id]
    
    def get_pending_requests(self) -> list[str]:
        """Get list of pending request IDs (queued or running).
        
        Returns:
            List of request IDs
        """
        with self._lock:
            return [
                req_id for req_id, req in self._queue.items()
                if req.status in ["queued", "submitted", "running"]
            ]
    
    def get_queue_depth(self) -> int:
        """Get current queue depth.
        
        Returns:
            Number of requests in queue
        """
        with self._lock:
            return len(self._queue)
    
    def clear_completed(self, max_age_seconds: float = 300.0) -> int:
        """Clear completed/failed requests older than max_age.
        
        Args:
            max_age_seconds: Maximum age in seconds (default: 5 minutes)
            
        Returns:
            Number of requests cleared
        """
        with self._lock:
            now = time.time()
            to_remove = []
            
            for request_id, request in self._queue.items():
                if request.status in ["completed", "failed"]:
                    age = now - request.submitted_at
                    if age > max_age_seconds:
                        to_remove.append(request_id)
            
            for request_id in to_remove:
                del self._queue[request_id]
            
            return len(to_remove)
    
    def clear(self) -> None:
        """Clear all requests from the queue."""
        with self._lock:
            self._queue.clear()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def __contains__(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._queue

