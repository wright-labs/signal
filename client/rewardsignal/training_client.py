"""Training-specialized client for Signal API."""

import requests
import time
from typing import List, Dict, Any, Optional
from collections import deque

from .schemas import (
    ForwardBackwardResponse,
    OptimStepResponse,
    SaveStateResponse,
)
from .exceptions import (
    SignalAPIError,
    ConnectionError as SignalConnectionError,
    TimeoutError as SignalTimeoutError,
)


class TrainingClient:
    """Specialized client for training operations with optimized defaults."""

    def __init__(
        self,
        run_id: str,
        api_key: str,
        base_url: str = "https://api.frontier-signal.com",
        timeout: int = 3600,  # 1 hour for training operations
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ):
        """Initialize training client."""
        self.run_id = run_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Use shared session if provided, otherwise create new one
        if session is not None:
            self.session = session
            self._owns_session = False
        else:
            self.session = requests.Session()
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            )
            self._owns_session = True

        # State tracking
        self.current_step = 0
        self.loss_history: deque = deque(maxlen=1000)  # Keep last 1000 losses
        self.grad_norm_history: deque = deque(maxlen=1000)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the session if we own it."""
        if self._owns_session and self.session:
            self.session.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a request with exponential backoff retry."""
        url = f"{self.base_url}{endpoint}"
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, json=json, timeout=self.timeout)

                if response.status_code >= 400:
                    # Let caller handle errors
                    response.raise_for_status()

                return response.json()

            except requests.exceptions.Timeout:
                last_exception = SignalTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout}s"
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = SignalConnectionError(f"Failed to connect to {url}: {str(e)}")
            except requests.exceptions.RequestException as e:
                last_exception = SignalAPIError(f"Request failed: {str(e)}")

            # Exponential backoff if not last attempt
            if attempt < self.max_retries - 1:
                wait_time = 2**attempt  # 1s, 2s, 4s
                time.sleep(wait_time)

        # All retries failed
        raise last_exception

    def forward_backward(
        self,
        batch_data: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ForwardBackwardResponse:
        """Compute gradients for a batch."""
        if loss_kwargs is None:
            loss_kwargs = {}

        payload = {
            "batch_data": batch_data,
            "accumulate": accumulate,
            "loss_fn": loss_fn,
            "loss_kwargs": loss_kwargs,
        }

        response_data = self._request("POST", f"/runs/{self.run_id}/forward_backward", json=payload)
        result = ForwardBackwardResponse(**response_data)

        # Track metrics
        self.loss_history.append(result.loss)
        if result.grad_norm is not None:
            self.grad_norm_history.append(result.grad_norm)

        return result

    def optim_step(
        self,
        learning_rate: Optional[float] = None,
    ) -> OptimStepResponse:
        """Apply optimizer update."""
        payload = {
            "learning_rate": learning_rate,
        }

        response_data = self._request("POST", f"/runs/{self.run_id}/optim_step", json=payload)
        result = OptimStepResponse(**response_data)

        # Update current step
        self.current_step = result.step

        return result

    def train_batch(
        self,
        batch_data: List[Dict[str, Any]],
        learning_rate: Optional[float] = None,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train on a single batch (convenience method)."""
        # Forward-backward
        fb_result = self.forward_backward(
            batch_data=batch_data,
            accumulate=False,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )

        # Optimizer step
        opt_result = self.optim_step(learning_rate=learning_rate)

        # Combine results
        return {
            "loss": fb_result.loss,
            "grad_norm": fb_result.grad_norm,
            "step": opt_result.step,
            "learning_rate": opt_result.learning_rate,
        }

    def train_epoch(
        self,
        dataloader,
        learning_rate: Optional[float] = None,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Train for one epoch over a dataloader."""
        epoch_losses = []
        epoch_grad_norms = []

        if progress:
            try:
                from tqdm import tqdm

                dataloader = tqdm(dataloader, desc="Training")
            except ImportError:
                pass  # No tqdm, just iterate normally

        for batch in dataloader:
            result = self.train_batch(
                batch_data=batch,
                learning_rate=learning_rate,
                loss_fn=loss_fn,
                loss_kwargs=loss_kwargs,
            )

            if "loss" in result:
                epoch_losses.append(result["loss"])
            if "grad_norm" in result:
                epoch_grad_norms.append(result["grad_norm"])

            # Update progress bar if using tqdm
            if progress and hasattr(dataloader, "set_postfix"):
                dataloader.set_postfix(
                    {
                        "loss": f"{result.get('loss', 0):.4f}",
                        "grad_norm": f"{result.get('grad_norm', 0):.4f}",
                    }
                )

        return {
            "num_batches": len(epoch_losses),
            "avg_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
            "avg_grad_norm": sum(epoch_grad_norms) / len(epoch_grad_norms)
            if epoch_grad_norms
            else 0,
            "final_step": self.current_step,
        }

    def save_checkpoint(
        self,
        mode: str = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> SaveStateResponse:
        """Save current model checkpoint."""
        payload = {
            "mode": mode,
            "push_to_hub": push_to_hub,
            "hub_model_id": hub_model_id,
        }

        response_data = self._request("POST", f"/runs/{self.run_id}/save_state", json=payload)
        return SaveStateResponse(**response_data)

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return {
            "current_step": self.current_step,
            "loss_history": list(self.loss_history),
            "grad_norm_history": list(self.grad_norm_history),
            "avg_loss": sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0,
            "avg_grad_norm": sum(self.grad_norm_history) / len(self.grad_norm_history)
            if self.grad_norm_history
            else 0,
        }
