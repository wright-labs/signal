"""Minimal wrapper around Modal's built-in futures.

This module provides a thin compatibility layer over Modal's native future support,
allowing for consistent API across sync and async clients.
"""
from typing import Any, Dict, Optional

# TODO: why do we have a future and an async client?

class SignalFuture:
    """Minimal wrapper around Modal future for consistent API.
    
    This class wraps Modal's native futures to provide a consistent interface
    for both sync and async operations while leveraging Modal's built-in
    queueing and execution management.
    
    Examples:
        >>> # With Modal
        >>> session = get_training_session(run_id)
        >>> modal_future = session.forward_backward.spawn(batch_data=batch, loss_fn="causal_lm")
        >>> future = SignalFuture(modal_future)
        >>> result = await future  # Async await
        >>> # or
        >>> result = future.result()  # Sync blocking
    """
    
    def __init__(self, modal_future: Any):
        """Wrap a Modal future.
        
        Args:
            modal_future: modal.Future from .spawn() call
        """
        self._modal_future = modal_future
        self._result: Optional[Dict[str, Any]] = None
        self._completed = False
    
    def __await__(self):
        """Await the future (async context).
        
        Returns:
            Awaitable that resolves to the result
        """
        return self._get_result().__await__()
    
    async def _get_result(self):
        """Internal async method to get result."""
        if not self._completed:
            # Modal's .get() is the blocking call
            self._result = self._modal_future.get()
            self._completed = True
        return self._result
    
    def result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get result synchronously (blocking).
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            
        Returns:
            Result dictionary from the operation
            
        Raises:
            TimeoutError: If timeout exceeded
        """
        if not self._completed:
            self._result = self._modal_future.get(timeout=timeout)
            self._completed = True
        return self._result
    
    def cancel(self) -> bool:
        """Cancel the future if possible.
        
        Returns:
            True if successfully cancelled, False otherwise
        """
        if self._completed:
            return False
        
        try:
            self._modal_future.cancel()
            return True
        except:
            return False
    
    def done(self) -> bool:
        """Check if the future is completed.
        
        Returns:
            True if completed, False otherwise
        """
        if self._completed:
            return True
        
        # Try to check if Modal future is done
        # Modal doesn't expose a done() method, so we try a very short timeout
        try:
            self._result = self._modal_future.get(timeout=0.001)
            self._completed = True
            return True
        except:
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        status = "completed" if self._completed else "pending"
        return f"<SignalFuture(status={status})>"


class FutureGroup:
    """Manage multiple futures as a group.
    
    Useful for batch operations and waiting on multiple futures.
    
    Examples:
        >>> group = FutureGroup()
        >>> for batch in batches:
        >>>     future = await client.forward_backward_async(batch, "causal_lm")
        >>>     group.add(future)
        >>> results = await group.wait_all()
    """
    
    def __init__(self):
        """Initialize empty future group."""
        self.futures: list[SignalFuture] = []
    
    def add(self, future: SignalFuture) -> None:
        """Add a future to the group.
        
        Args:
            future: SignalFuture to add
        """
        self.futures.append(future)
    
    async def wait_all(self, return_exceptions: bool = False) -> list[Dict[str, Any]]:
        """Wait for all futures to complete.
        
        Args:
            return_exceptions: If True, exceptions are returned instead of raised
            
        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        import asyncio
        
        if return_exceptions:
            results = await asyncio.gather(
                *[future._get_result() for future in self.futures],
                return_exceptions=True
            )
        else:
            results = await asyncio.gather(
                *[future._get_result() for future in self.futures]
            )
        
        return results
    
    def cancel_all(self) -> None:
        """Cancel all futures in the group."""
        for future in self.futures:
            future.cancel()
    
    def __len__(self) -> int:
        """Get number of futures in group."""
        return len(self.futures)
    
    def __iter__(self):
        """Iterate over futures."""
        return iter(self.futures)
