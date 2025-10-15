"""Server-side request orchestrator for managing queued requests.

This module handles request queueing, ordering, and status tracking
for the futures-based request pipelining system.
"""

import time
import uuid
from typing import Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import asyncio


@dataclass
class ServerRequest:
    """Represents a server-side queued request."""
    request_id: str
    run_id: str
    request_type: str  # "forward_backward", "optim_step", etc.
    payload: Dict[str, Any]
    submitted_at: float = field(default_factory=time.time)
    status: str = "queued"  # "queued", "running", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class RequestOrchestrator:
    """Server-side request orchestrator.
    
    Manages queued requests, ensures ordering guarantees, and tracks
    request status for futures-based clients.
    
    Thread-safe for concurrent access.
    """
    
    def __init__(self, max_queue_size_per_run: int = 50):
        """Initialize request orchestrator.
        
        Args:
            max_queue_size_per_run: Max queued requests per run
        """
        self.max_queue_size_per_run = max_queue_size_per_run
        
        # Per-run request queues: run_id -> OrderedDict[request_id -> ServerRequest]
        self._queues: Dict[str, OrderedDict[str, ServerRequest]] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Background cleanup thread
        self._cleanup_running = False
        self._cleanup_thread: Optional[threading.Thread] = None
    
    def start_cleanup_thread(self, interval: float = 60.0):
        """Start background cleanup thread.
        
        Args:
            interval: Cleanup interval in seconds
        """
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval,),
            daemon=True,
        )
        self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def _cleanup_loop(self, interval: float):
        """Background cleanup loop."""
        while self._cleanup_running:
            try:
                self.cleanup_old_requests(max_age_seconds=600)  # 10 minutes
                time.sleep(interval)
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"req_{uuid.uuid4().hex[:16]}"
    
    def submit_request(
        self,
        run_id: str,
        request_type: str,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> str:
        """Submit a new request.
        
        Args:
            run_id: Run identifier
            request_type: Type of request
            payload: Request payload
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            Request ID
            
        Raises:
            RuntimeError: If queue is full for this run
        """
        with self._lock:
            # Create queue for run if doesn't exist
            if run_id not in self._queues:
                self._queues[run_id] = OrderedDict()
            
            queue = self._queues[run_id]
            
            # Check queue size
            if len(queue) >= self.max_queue_size_per_run:
                raise RuntimeError(
                    f"Request queue full for run {run_id} "
                    f"({self.max_queue_size_per_run} requests)"
                )
            
            # Generate request ID if not provided
            if request_id is None:
                request_id = self.generate_request_id()
            
            # Create request
            request = ServerRequest(
                request_id=request_id,
                run_id=run_id,
                request_type=request_type,
                payload=payload,
                status="queued",
            )
            
            # Add to queue
            queue[request_id] = request
            
            return request_id
    
    def get_request_status(
        self,
        run_id: str,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the status of a request.
        
        Args:
            run_id: Run identifier
            request_id: Request ID
            
        Returns:
            Status dictionary or None if not found
        """
        with self._lock:
            if run_id not in self._queues:
                return None
            
            queue = self._queues[run_id]
            if request_id not in queue:
                return None
            
            request = queue[request_id]
            return {
                "request_id": request.request_id,
                "run_id": request.run_id,
                "request_type": request.request_type,
                "status": request.status,
                "result": request.result,
                "error": request.error,
                "submitted_at": request.submitted_at,
                "started_at": request.started_at,
                "completed_at": request.completed_at,
            }
    
    def update_request_status(
        self,
        run_id: str,
        request_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the status of a request.
        
        Args:
            run_id: Run identifier
            request_id: Request ID
            status: New status
            result: Optional result data
            error: Optional error message
        """
        with self._lock:
            if run_id not in self._queues:
                return
            
            queue = self._queues[run_id]
            if request_id not in queue:
                return
            
            request = queue[request_id]
            request.status = status
            
            if status == "running" and request.started_at is None:
                request.started_at = time.time()
            
            if status in ["completed", "failed"]:
                request.completed_at = time.time()
            
            if result is not None:
                request.result = result
            
            if error is not None:
                request.error = error
    
    def get_next_queued_request(
        self,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the next queued request for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Request dictionary or None if no queued requests
        """
        with self._lock:
            if run_id not in self._queues:
                return None
            
            queue = self._queues[run_id]
            
            # Find first queued request
            for request_id, request in queue.items():
                if request.status == "queued":
                    # Mark as running
                    request.status = "running"
                    request.started_at = time.time()
                    
                    return {
                        "request_id": request.request_id,
                        "request_type": request.request_type,
                        "payload": request.payload,
                    }
            
            return None
    
    def get_queue_depth(self, run_id: str) -> Dict[str, int]:
        """Get queue depth statistics for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with counts by status
        """
        with self._lock:
            if run_id not in self._queues:
                return {
                    "total": 0,
                    "queued": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0,
                }
            
            queue = self._queues[run_id]
            
            counts = {
                "total": len(queue),
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
            }
            
            for request in queue.values():
                if request.status in counts:
                    counts[request.status] += 1
            
            return counts
    
    def remove_request(self, run_id: str, request_id: str) -> None:
        """Remove a request from the queue.
        
        Args:
            run_id: Run identifier
            request_id: Request ID
        """
        with self._lock:
            if run_id in self._queues:
                queue = self._queues[run_id]
                if request_id in queue:
                    del queue[request_id]
    
    def cleanup_old_requests(
        self,
        max_age_seconds: float = 600.0,
    ) -> int:
        """Clean up old completed/failed requests.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of requests cleaned up
        """
        with self._lock:
            now = time.time()
            total_removed = 0
            
            for run_id, queue in list(self._queues.items()):
                to_remove = []
                
                for request_id, request in queue.items():
                    if request.status in ["completed", "failed"]:
                        age = now - request.submitted_at
                        if age > max_age_seconds:
                            to_remove.append(request_id)
                
                for request_id in to_remove:
                    del queue[request_id]
                    total_removed += 1
                
                # Remove empty queues
                if len(queue) == 0:
                    del self._queues[run_id]
            
            return total_removed
    
    def clear_run_queue(self, run_id: str) -> int:
        """Clear all requests for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Number of requests cleared
        """
        with self._lock:
            if run_id in self._queues:
                count = len(self._queues[run_id])
                del self._queues[run_id]
                return count
            return 0
    
    def get_all_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues.
        
        Returns:
            Dictionary with global statistics
        """
        with self._lock:
            total_requests = sum(len(queue) for queue in self._queues.values())
            
            stats = {
                "total_runs": len(self._queues),
                "total_requests": total_requests,
                "queues": {},
            }
            
            for run_id, queue in self._queues.items():
                stats["queues"][run_id] = self.get_queue_depth(run_id)
            
            return stats


# Global orchestrator instance
_global_orchestrator: Optional[RequestOrchestrator] = None


def get_orchestrator() -> RequestOrchestrator:
    """Get the global request orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = RequestOrchestrator()
        _global_orchestrator.start_cleanup_thread()
    return _global_orchestrator

