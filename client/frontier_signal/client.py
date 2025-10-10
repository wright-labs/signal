"""Synchronous Python client SDK for Signal API."""

import requests
from typing import List, Dict, Any, Optional, Literal

from .schemas import (
    RunConfig,
    RunResponse,
    RunStatus,
    RunMetrics,
    TrainingExample,
    ForwardBackwardResponse,
    OptimStepResponse,
    SampleResponse,
    SaveStateResponse,
)
from .exceptions import (
    SignalAPIError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    ConnectionError as SignalConnectionError,
    TimeoutError as SignalTimeoutError,
)


class SignalRun:
    """Represents a training run with convenient methods."""
    
    def __init__(self, client: "SignalClient", run_id: str, config: Dict[str, Any]):
        """Initialize a training run.
        
        Args:
            client: SignalClient instance
            run_id: Run identifier
            config: Run configuration
        """
        self.client = client
        self.run_id = run_id
        self.config = config
    
    def forward_backward(
        self,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        **loss_kwargs
    ) -> Dict[str, Any]:
        """Perform forward-backward pass.
        
        Args:
            batch: List of training examples
            accumulate: Whether to accumulate gradients
            loss_fn: Loss function to use (causal_lm, dpo, reward_modeling, ppo)
            **loss_kwargs: Additional arguments for the loss function
            
        Returns:
            Response with loss and gradient stats
        """
        return self.client.forward_backward(
            run_id=self.run_id,
            batch=batch,
            accumulate=accumulate,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )
    
    def optim_step(
        self,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply optimizer step.
        
        Args:
            learning_rate: Optional learning rate override
            
        Returns:
            Response with step metrics
        """
        return self.client.optim_step(
            run_id=self.run_id,
            learning_rate=learning_rate,
        )
    
    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Generate samples.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Response with generated outputs
        """
        return self.client.sample(
            run_id=self.run_id,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=return_logprobs,
        )
    
    def save_state(
        self,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model state.
        
        Args:
            mode: Save mode ('adapter' or 'merged')
            push_to_hub: Whether to push to HuggingFace Hub
            hub_model_id: HuggingFace Hub model ID
            
        Returns:
            Response with artifact information
        """
        return self.client.save_state(
            run_id=self.run_id,
            mode=mode,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get run status.
        
        Returns:
            Run status information
        """
        return self.client.get_run_status(self.run_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get run metrics.
        
        Returns:
            Run metrics
        """
        return self.client.get_run_metrics(self.run_id)


class SignalClient:
    """Synchronous client for Signal API."""
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://api.frontier-signal.com",
        timeout: int = 300,
    ):
        """Initialize the client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the API server
            timeout: Request timeout in seconds (default: 300)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def _handle_error(self, response: requests.Response) -> None:
        """Handle error responses.
        
        Args:
            response: HTTP response
            
        Raises:
            SignalAPIError: Appropriate exception based on status code
        """
        try:
            error_data = response.json()
            error_msg = error_data.get("detail") or error_data.get("error", response.text)
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code} error"
        
        status_code = response.status_code
        
        if status_code == 401:
            raise AuthenticationError(error_msg, response_data=error_data if 'error_data' in locals() else None)
        elif status_code == 403:
            raise AuthorizationError(error_msg, response_data=error_data if 'error_data' in locals() else None)
        elif status_code == 404:
            raise NotFoundError(error_msg, response_data=error_data if 'error_data' in locals() else None)
        elif status_code == 422:
            raise ValidationError(error_msg, response_data=error_data if 'error_data' in locals() else None)
        elif status_code == 429:
            raise RateLimitError(error_msg, response_data=error_data if 'error_data' in locals() else None)
        elif status_code >= 500:
            raise ServerError(error_msg, status_code=status_code, response_data=error_data if 'error_data' in locals() else None)
        else:
            raise SignalAPIError(error_msg, status_code=status_code, response_data=error_data if 'error_data' in locals() else None)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a request to the API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            json: Optional JSON payload
            
        Returns:
            Response data
            
        Raises:
            SignalAPIError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, 
                url, 
                json=json,
                timeout=self.timeout
            )
        except requests.exceptions.Timeout:
            raise SignalTimeoutError(f"Request to {endpoint} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise SignalConnectionError(f"Failed to connect to {url}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise SignalAPIError(f"Request failed: {str(e)}")
        
        if response.status_code >= 400:
            self._handle_error(response)
        
        return response.json()
    
    def list_models(self) -> List[str]:
        """List available models.
        
        Returns:
            List of model names
        """
        response = self._request("GET", "/models")
        return response["models"]
    
    def create_run(
        self,
        base_model: str,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        optimizer: str = "adamw_8bit",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        bf16: bool = True,
        gradient_checkpointing: bool = True,
    ) -> SignalRun:
        """Create a new training run.
        
        Args:
            base_model: Base model name
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: Target modules for LoRA
            optimizer: Optimizer type
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_seq_length: Maximum sequence length
            bf16: Use bfloat16
            gradient_checkpointing: Enable gradient checkpointing
            
        Returns:
            SignalRun instance
        """
        config = RunConfig(
            base_model=base_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_seq_length=max_seq_length,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        response = self._request("POST", "/runs", json=config.model_dump())
        
        return SignalRun(
            client=self,
            run_id=response["run_id"],
            config=response["config"],
        )
    
    def forward_backward(
        self,
        run_id: str,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform forward-backward pass.
        
        Args:
            run_id: Run identifier
            batch: List of training examples
            accumulate: Whether to accumulate gradients
            loss_fn: Loss function to use (causal_lm, dpo, reward_modeling, ppo)
            loss_kwargs: Additional arguments for the loss function
            
        Returns:
            Response with loss and gradient stats
        """
        if loss_kwargs is None:
            loss_kwargs = {}
            
        payload = {
            "batch_data": batch,
            "accumulate": accumulate,
            "loss_fn": loss_fn,
            "loss_kwargs": loss_kwargs,
        }
        
        return self._request("POST", f"/runs/{run_id}/forward_backward", json=payload)
    
    def optim_step(
        self,
        run_id: str,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply optimizer step.
        
        Args:
            run_id: Run identifier
            learning_rate: Optional learning rate override
            
        Returns:
            Response with step metrics
        """
        payload = {
            "learning_rate": learning_rate,
        }
        
        return self._request("POST", f"/runs/{run_id}/optim_step", json=payload)
    
    def sample(
        self,
        run_id: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Generate samples.
        
        Args:
            run_id: Run identifier
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Response with generated outputs
        """
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_logprobs": return_logprobs,
        }
        
        return self._request("POST", f"/runs/{run_id}/sample", json=payload)
    
    def save_state(
        self,
        run_id: str,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save model state.
        
        Args:
            run_id: Run identifier
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
        
        return self._request("POST", f"/runs/{run_id}/save_state", json=payload)
    
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get run status.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run status information
        """
        return self._request("GET", f"/runs/{run_id}/status")
    
    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get run metrics.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run metrics
        """
        return self._request("GET", f"/runs/{run_id}/metrics")
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs.
        
        Returns:
            List of runs
        """
        response = self._request("GET", "/runs")
        return response["runs"]
