"""Futures implementation for Signal API with double-await pattern.

This module implements Tinker-style futures that enable request pipelining:
- First await: Submits request and returns future (ordering guaranteed)
- Second await: Waits for computation to complete and returns result

This allows overlapping request submission with execution for better throughput.
"""

import asyncio
import time
from typing import Any, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .async_training_client_v2 import AsyncTrainingClientV2


@dataclass
class RequestStatus:
    """Status of a request in the processing pipeline."""
    request_id: str
    status: str  # "queued", "running", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    submitted_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class SignalFuture:
    """Future representing a pending training operation.
    
    Implements the double-await pattern:
    
    Example:
        # First await: submit request (non-blocking)
        future = await client.forward_backward_async(batch, "ppo")
        
        # Can submit more requests here while first executes
        future2 = await client.forward_backward_async(batch2, "ppo")
        
        # Second await: wait for completion (blocking)
        result1 = await future
        result2 = await future2
    """
    
    def __init__(
        self,
        request_id: str,
        client: "AsyncTrainingClientV2",
        poll_interval: float = 0.1,
        timeout: Optional[float] = None,
    ):
        """Initialize future.
        
        Args:
            request_id: Unique identifier for this request
            client: Client instance to poll for status
            poll_interval: Seconds between status polls (default: 0.1)
            timeout: Maximum seconds to wait (default: None = no timeout)
        """
        self.request_id = request_id
        self._client = client
        self._poll_interval = poll_interval
        self._timeout = timeout
        
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._completed = False
        self._cancelled = False
        self._start_time = time.time()
    
    def __await__(self):
        """Await the future to get the result.
        
        This implements the second await in the double-await pattern.
        Polls the server until the request completes.
        """
        return self._wait_for_completion().__await__()
    
    async def _wait_for_completion(self) -> Dict[str, Any]:
        """Poll for completion and return result."""
        if self._completed:
            if self._error:
                raise RuntimeError(f"Request {self.request_id} failed: {self._error}")
            return self._result
        
        if self._cancelled:
            raise asyncio.CancelledError(f"Request {self.request_id} was cancelled")
        
        # Poll for completion
        while not self._completed and not self._cancelled:
            # Check timeout
            if self._timeout and (time.time() - self._start_time) > self._timeout:
                self.cancel()
                raise asyncio.TimeoutError(
                    f"Request {self.request_id} timed out after {self._timeout}s"
                )
            
            try:
                # Check status from server
                status = await self._client._check_future_status(self.request_id)
                
                if status["status"] == "completed":
                    self._result = status["result"]
                    self._completed = True
                    return self._result
                
                elif status["status"] == "failed":
                    self._error = status.get("error", "Unknown error")
                    self._completed = True
                    raise RuntimeError(f"Request {self.request_id} failed: {self._error}")
                
                elif status["status"] in ["queued", "running"]:
                    # Still processing, wait and poll again
                    await asyncio.sleep(self._poll_interval)
                
                else:
                    raise ValueError(f"Unknown status: {status['status']}")
                    
            except asyncio.CancelledError:
                self._cancelled = True
                raise
            except Exception as e:
                self._error = str(e)
                self._completed = True
                raise
        
        # If we get here, request was cancelled
        if self._cancelled:
            raise asyncio.CancelledError(f"Request {self.request_id} was cancelled")
        
        return self._result
    
    def cancel(self) -> bool:
        """Cancel the request if still in progress.
        
        Returns:
            True if request was cancelled, False if already completed
        """
        if self._completed:
            return False
        
        self._cancelled = True
        # TODO: Send cancellation request to server
        return True
    
    def done(self) -> bool:
        """Check if the future is done (completed or failed)."""
        return self._completed
    
    def cancelled(self) -> bool:
        """Check if the future was cancelled."""
        return self._cancelled
    
    def result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get the result synchronously (blocks).
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            Result dictionary
            
        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If request failed
            CancelledError: If request was cancelled
        """
        if self._completed:
            if self._error:
                raise RuntimeError(f"Request {self.request_id} failed: {self._error}")
            return self._result
        
        if self._cancelled:
            raise asyncio.CancelledError(f"Request {self.request_id} was cancelled")
        
        # Run async wait in sync context
        loop = asyncio.get_event_loop()
        if timeout:
            return loop.run_until_complete(
                asyncio.wait_for(self._wait_for_completion(), timeout=timeout)
            )
        else:
            return loop.run_until_complete(self._wait_for_completion())
    
    def __repr__(self) -> str:
        status = "cancelled" if self._cancelled else "completed" if self._completed else "pending"
        return f"<SignalFuture(request_id={self.request_id}, status={status})>"


class FutureGroup:
    """Manage multiple futures as a group.
    
    Useful for batch operations and pipelining.
    
    Example:
        group = FutureGroup()
        
        # Submit multiple requests
        for batch in dataloader:
            future = await client.forward_backward_async(batch, "ppo")
            group.add(future)
        
        # Wait for all to complete
        results = await group.wait_all()
    """
    
    def __init__(self):
        self.futures: list[SignalFuture] = []
    
    def add(self, future: SignalFuture) -> None:
        """Add a future to the group."""
        self.futures.append(future)
    
    async def wait_all(self, return_exceptions: bool = False) -> list[Dict[str, Any]]:
        """Wait for all futures to complete.
        
        Args:
            return_exceptions: If True, exceptions are returned instead of raised
            
        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        if return_exceptions:
            results = await asyncio.gather(
                *[future._wait_for_completion() for future in self.futures],
                return_exceptions=True
            )
        else:
            results = await asyncio.gather(
                *[future._wait_for_completion() for future in self.futures]
            )
        
        return results
    
    async def wait_any(self) -> Dict[str, Any]:
        """Wait for the first future to complete.
        
        Returns:
            Result from the first completed future
        """
        done, pending = await asyncio.wait(
            [future._wait_for_completion() for future in self.futures],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Get result from first completed
        result = await next(iter(done))
        
        # Cancel pending futures
        for task in pending:
            task.cancel()
        
        return result
    
    def cancel_all(self) -> None:
        """Cancel all futures in the group."""
        for future in self.futures:
            future.cancel()
    
    def __len__(self) -> int:
        return len(self.futures)
    
    def __iter__(self):
        return iter(self.futures)


def create_pipelined_batch_processor(
    max_concurrent: int = 3,
) -> callable:
    """Create a semaphore-based batch processor for pipelining.
    
    This limits the number of concurrent in-flight requests to prevent
    overwhelming the server while still allowing pipelining.
    
    Args:
        max_concurrent: Maximum concurrent requests
        
    Returns:
        Async context manager function
        
    Example:
        processor = create_pipelined_batch_processor(max_concurrent=3)
        
        for batch in dataloader:
            async with processor():
                future = await client.forward_backward_async(batch, "ppo")
                results.append(await future)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def processor():
        async with semaphore:
            yield
    
    return processor

