"""Async training client V2 with futures support and request pipelining.

This client implements Tinker's double-await pattern for optimal request pipelining:
- First await: Submits request to server queue (non-blocking)
- Second await: Waits for computation to complete (blocking)

Enable with environment variable: SIGNAL_ENABLE_FUTURES=true
"""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from collections import deque

from .futures import SignalFuture, FutureGroup
from .request_queue import ClientRequestQueue
from .exceptions import (
    SignalAPIError,
    ConnectionError as SignalConnectionError,
    TimeoutError as SignalTimeoutError,
)


class AsyncTrainingClientV2:
    """Async training client with futures support for request pipelining.
    
    This client uses the double-await pattern:
    
    Example:
        # Create client
        client = AsyncTrainingClientV2(run_id, api_key)
        
        # Submit multiple requests (non-blocking)
        future1 = await client.forward_backward_async(batch1, "ppo")
        future2 = await client.forward_backward_async(batch2, "ppo")  # Queued while first runs
        future3 = await client.forward_backward_async(batch3, "ppo")
        
        # Wait for results (blocking)
        result1 = await future1
        result2 = await future2
        result3 = await future3
    """
    
    def __init__(
        self,
        run_id: str,
        api_key: str,
        base_url: str = "https://api.frontier-signal.com",
        timeout: int = 3600,
        max_retries: int = 3,
        session: Optional[aiohttp.ClientSession] = None,
        enable_futures: Optional[bool] = None,
        max_concurrent_requests: int = 3,
    ):
        """Initialize async training client V2.
        
        Args:
            run_id: Run identifier
            api_key: API key for authentication
            base_url: Base URL of the API server
            timeout: Request timeout in seconds (default: 3600 = 1 hour)
            max_retries: Number of retries for failed requests (default: 3)
            session: Optional shared session (for connection pooling)
            enable_futures: Enable futures mode (default: read from env SIGNAL_ENABLE_FUTURES)
            max_concurrent_requests: Max concurrent in-flight requests (default: 3)
        """
        self.run_id = run_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Determine if futures mode is enabled
        if enable_futures is None:
            enable_futures = os.getenv("SIGNAL_ENABLE_FUTURES", "false").lower() == "true"
        self.enable_futures = enable_futures
        
        # Use shared session if provided
        if session is not None:
            self.session = session
            self._owns_session = False
        else:
            self.session = None
            self._owns_session = True
        
        # Request queue for futures mode
        self.request_queue = ClientRequestQueue(max_queue_size=100)
        self.max_concurrent_requests = max_concurrent_requests
        self._concurrent_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # State tracking
        self.current_step = 0
        self.loss_history: deque = deque(maxlen=1000)
        self.grad_norm_history: deque = deque(maxlen=1000)
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self._owns_session and self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the session if we own it."""
        if self._owns_session and self.session:
            await self.session.close()
            self.session = None
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an async request with exponential backoff retry.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            json: Optional JSON payload
            
        Returns:
            Response data
            
        Raises:
            SignalAPIError: If request fails after retries
        """
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
            self._owns_session = True
        
        url = f"{self.base_url}{endpoint}"
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with self.session.request(
                    method,
                    url,
                    json=json,
                    timeout=timeout
                ) as response:
                    if response.status >= 400:
                        response.raise_for_status()
                    
                    return await response.json()
                    
            except asyncio.TimeoutError as e:
                last_exception = SignalTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout}s"
                )
            except aiohttp.ClientConnectionError as e:
                last_exception = SignalConnectionError(
                    f"Failed to connect to {url}: {str(e)}"
                )
            except Exception as e:
                last_exception = SignalAPIError(f"Request failed: {str(e)}")
            
            # Exponential backoff if not last attempt
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        # All retries failed
        raise last_exception
    
    async def forward_backward_async(
        self,
        batch_data: List[Dict[str, Any]],
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        accumulate: bool = False,
    ) -> SignalFuture:
        """Compute gradients for a batch (async, returns future).
        
        This is the first await in the double-await pattern.
        The request is submitted and a future is returned immediately.
        
        Args:
            batch_data: List of training examples
            loss_fn: Loss function to use
            loss_kwargs: Additional arguments for loss function
            accumulate: Whether to accumulate gradients
            
        Returns:
            SignalFuture that can be awaited for the result
        """
        if loss_kwargs is None:
            loss_kwargs = {}
        
        payload = {
            "batch_data": batch_data,
            "accumulate": accumulate,
            "loss_fn": loss_fn,
            "loss_kwargs": loss_kwargs,
        }
        
        if self.enable_futures:
            # Futures mode: Submit request and return future
            request_id = self.request_queue.submit(
                request_type="forward_backward",
                payload=payload,
            )
            
            # Add request_id to payload for server tracking
            payload["request_id"] = request_id
            
            # Submit request to server (non-blocking)
            # Use semaphore to limit concurrent requests
            async with self._concurrent_semaphore:
                response = await self._request(
                    "POST",
                    f"/runs/{self.run_id}/forward_backward",
                    json=payload
                )
                
                # Update queue status
                if response.get("status") == "queued":
                    self.request_queue.update_status(request_id, "submitted")
                
                # Return future
                return SignalFuture(
                    request_id=request_id,
                    client=self,
                    timeout=self.timeout,
                )
        else:
            # Legacy mode: Synchronous request
            result = await self._request(
                "POST",
                f"/runs/{self.run_id}/forward_backward",
                json=payload
            )
            
            # Track metrics
            if "loss" in result:
                self.loss_history.append(result["loss"])
            if "grad_norm" in result:
                self.grad_norm_history.append(result["grad_norm"])
            
            # Return a completed future for API consistency
            request_id = self.request_queue.generate_request_id()
            self.request_queue.submit(
                request_type="forward_backward",
                payload=payload,
                request_id=request_id,
            )
            self.request_queue.update_status(request_id, "completed", result=result)
            
            future = SignalFuture(request_id=request_id, client=self)
            future._completed = True
            future._result = result
            return future
    
    async def optim_step_async(
        self,
        learning_rate: Optional[float] = None,
    ) -> SignalFuture:
        """Apply optimizer update (async, returns future).
        
        This is the first await in the double-await pattern.
        
        Args:
            learning_rate: Optional learning rate override
            
        Returns:
            SignalFuture that can be awaited for the result
        """
        payload = {
            "learning_rate": learning_rate,
        }
        
        if self.enable_futures:
            # Futures mode: Submit request and return future
            request_id = self.request_queue.submit(
                request_type="optim_step",
                payload=payload,
            )
            
            payload["request_id"] = request_id
            
            # Submit request to server
            async with self._concurrent_semaphore:
                response = await self._request(
                    "POST",
                    f"/runs/{self.run_id}/optim_step",
                    json=payload
                )
                
                if response.get("status") == "queued":
                    self.request_queue.update_status(request_id, "submitted")
                
                return SignalFuture(
                    request_id=request_id,
                    client=self,
                    timeout=self.timeout,
                )
        else:
            # Legacy mode: Synchronous request
            result = await self._request(
                "POST",
                f"/runs/{self.run_id}/optim_step",
                json=payload
            )
            
            if "step" in result:
                self.current_step = result["step"]
            
            request_id = self.request_queue.generate_request_id()
            self.request_queue.submit(
                request_type="optim_step",
                payload=payload,
                request_id=request_id,
            )
            self.request_queue.update_status(request_id, "completed", result=result)
            
            future = SignalFuture(request_id=request_id, client=self)
            future._completed = True
            future._result = result
            return future
    
    async def _check_future_status(self, request_id: str) -> Dict[str, Any]:
        """Check the status of a future request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Status dictionary with 'status', 'result', 'error'
        """
        # Check local queue first
        local_status = self.request_queue.get_status(request_id)
        if local_status and local_status["status"] == "completed":
            return {
                "status": "completed",
                "result": local_status["result"],
            }
        elif local_status and local_status["status"] == "failed":
            return {
                "status": "failed",
                "error": local_status["error"],
            }
        
        # Query server for status
        try:
            response = await self._request(
                "GET",
                f"/runs/{self.run_id}/requests/{request_id}/status",
            )
            
            # Update local queue
            status = response.get("status", "unknown")
            if status == "completed":
                self.request_queue.update_status(
                    request_id,
                    "completed",
                    result=response.get("result"),
                )
            elif status == "failed":
                self.request_queue.update_status(
                    request_id,
                    "failed",
                    error=response.get("error"),
                )
            
            return response
            
        except Exception as e:
            # If server query fails, return error
            return {
                "status": "failed",
                "error": f"Failed to check status: {str(e)}",
            }
    
    async def train_batch(
        self,
        batch_data: List[Dict[str, Any]],
        learning_rate: Optional[float] = None,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train on a single batch (async convenience method with futures).
        
        Args:
            batch_data: List of training examples
            learning_rate: Optional learning rate override
            loss_fn: Loss function to use
            loss_kwargs: Additional arguments for loss function
            
        Returns:
            Combined result with loss, grad_norm, and step
        """
        # Submit both requests
        fb_future = await self.forward_backward_async(
            batch_data=batch_data,
            accumulate=False,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )
        opt_future = await self.optim_step_async(learning_rate=learning_rate)
        
        # Wait for results
        fb_result = await fb_future
        opt_result = await opt_future
        
        # Combine results
        return {
            "loss": fb_result.get("loss"),
            "grad_norm": fb_result.get("grad_norm"),
            "step": opt_result.get("step"),
            "learning_rate": opt_result.get("learning_rate"),
        }
    
    async def save_checkpoint(
        self,
        mode: str = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save current model checkpoint (async).
        
        Args:
            mode: Save mode ('adapter' or 'merged')
            push_to_hub: Whether to push to HuggingFace Hub
            hub_model_id: HuggingFace Hub model ID
            
        Returns:
            Response with artifact information
        """
        payload = {
            "mode": mode,
            "push_to_hub": push_to_hub,
            "hub_model_id": hub_model_id,
        }
        
        return await self._request(
            "POST",
            f"/runs/{self.run_id}/save_state",
            json=payload
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics (sync).
        
        Returns:
            Dict with loss_history, grad_norm_history, current_step
        """
        return {
            "current_step": self.current_step,
            "loss_history": list(self.loss_history),
            "grad_norm_history": list(self.grad_norm_history),
            "avg_loss": sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0,
            "avg_grad_norm": sum(self.grad_norm_history) / len(self.grad_norm_history) if self.grad_norm_history else 0,
            "queue_depth": self.request_queue.get_queue_depth(),
            "pending_requests": len(self.request_queue.get_pending_requests()),
        }
    
    def create_future_group(self) -> FutureGroup:
        """Create a future group for batch operations.
        
        Returns:
            FutureGroup instance
        """
        return FutureGroup()

