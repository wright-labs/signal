"""Asynchronous Python client SDK for Signal API."""

import httpx
from typing import List, Dict, Any, Optional, Literal, Union

from .schemas import (
    RunConfig,
    RunResponse,
    RunStatus,
    RunMetrics,
    ForwardBackwardResponse,
    OptimStepResponse,
    SampleResponse,
    SaveStateResponse,
    TokenizeResponse,
    DetokenizeResponse,
    TokenizerInfoResponse,
    ModelInfoResponse,
    ApplyChatTemplateResponse,
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
from .futures import APIFuture


class AsyncSignalRun:
    """Represents a training run with convenient async methods."""
    
    def __init__(self, client: "AsyncSignalClient", run_id: str, config: Dict[str, Any]):
        """Initialize a training run."""
        self.client = client
        self.run_id = run_id
        self.config = config
    
    async def forward_backward(
        self,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        **loss_kwargs
    ) -> ForwardBackwardResponse:
        """Perform forward-backward pass (blocking)."""
        return await self.client.forward_backward(
            run_id=self.run_id,
            batch=batch,
            accumulate=accumulate,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )
    
    async def optim_step(
        self,
        learning_rate: Optional[float] = None,
    ) -> OptimStepResponse:
        """Apply optimizer step."""
        return await self.client.optim_step(
            run_id=self.run_id,
            learning_rate=learning_rate,
        )
    
    async def sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> SampleResponse:
        """Generate samples."""
        return await self.client.sample(
            run_id=self.run_id,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=return_logprobs,
        )
    
    async def save_state(
        self,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> SaveStateResponse:
        """Save model state."""
        return await self.client.save_state(
            run_id=self.run_id,
            mode=mode,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
    
    async def forward_backward_async(
        self,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        **loss_kwargs
    ) -> APIFuture:
        """Perform forward-backward pass (async)."""
        return await self.client.forward_backward_async(
            run_id=self.run_id,
            batch=batch,
            accumulate=accumulate,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )
    
    async def optim_step_async(
        self,
        learning_rate: Optional[float] = None,
    ) -> APIFuture:
        """Apply optimizer step (async)."""
        return await self.client.optim_step_async(
            run_id=self.run_id,
            learning_rate=learning_rate,
        )
    
    async def sample_async(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> APIFuture:
        """Generate samples (async)."""
        return await self.client.sample_async(
            run_id=self.run_id,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=return_logprobs,
        )
    
    async def save_state_async(
        self,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> APIFuture:
        """Save model state (async)."""
        return await self.client.save_state_async(
            run_id=self.run_id,
            mode=mode,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
    
    async def get_status(self) -> RunStatus:
        """Get run status."""
        return await self.client.get_run_status(self.run_id)
    
    async def get_metrics(self) -> RunMetrics:
        """Get run metrics."""
        return await self.client.get_run_metrics(self.run_id)
    
    async def tokenize(
        self,
        text: str | List[str],
        add_special_tokens: bool = True,
    ) -> TokenizeResponse:
        """Tokenize text using the model's tokenizer."""
        return await self.client.tokenize(
            run_id=self.run_id,
            text=text,
            add_special_tokens=add_special_tokens,
        )
    
    async def detokenize(
        self,
        token_ids: List[int] | List[List[int]],
    ) -> DetokenizeResponse:
        """Detokenize token IDs using the model's tokenizer."""
        return await self.client.detokenize(
            run_id=self.run_id,
            token_ids=token_ids,
        )
    
    async def get_tokenizer_info(self) -> TokenizerInfoResponse:
        """Get tokenizer configuration information."""
        return await self.client.get_tokenizer_info(self.run_id)
    
    async def get_model_info(self) -> ModelInfoResponse:
        """Get model architecture information."""
        return await self.client.get_model_info(self.run_id)
    
    async def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> ApplyChatTemplateResponse:
        """Apply the model's chat template to format messages."""
        return await self.client.apply_chat_template(
            run_id=self.run_id,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )


class AsyncSignalClient:
    """Asynchronous client for Signal API."""
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://signal-production-d2d8.up.railway.app",
        timeout: int = 300,
    ):
        """Initialize the async client."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client
    
    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses."""
        # Parse error data once, default to None if parsing fails
        error_data = None
        try:
            error_data = response.json()
            error_msg = error_data.get("detail") or error_data.get("error", response.text)
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code} error"
        
        status_code = response.status_code
        
        if status_code == 401:
            raise AuthenticationError(error_msg, response_data=error_data)
        elif status_code == 403:
            raise AuthorizationError(error_msg, response_data=error_data)
        elif status_code == 404:
            raise NotFoundError(error_msg, response_data=error_data)
        elif status_code == 422:
            raise ValidationError(error_msg, response_data=error_data)
        elif status_code == 429:
            raise RateLimitError(error_msg, response_data=error_data)
        elif status_code >= 500:
            raise ServerError(error_msg, status_code=status_code, response_data=error_data)
        else:
            raise SignalAPIError(error_msg, status_code=status_code, response_data=error_data)
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a request to the API."""
        url = f"{self.base_url}{endpoint}"
        client = self._get_client()
        
        try:
            response = await client.request(
                method, 
                url, 
                json=json,
            )
        except httpx.TimeoutException:
            raise SignalTimeoutError(f"Request to {endpoint} timed out after {self.timeout}s")
        except httpx.ConnectError as e:
            raise SignalConnectionError(f"Failed to connect to {url}: {str(e)}")
        except httpx.RequestError as e:
            raise SignalAPIError(f"Request failed: {str(e)}")
        
        if response.status_code >= 400:
            self._handle_error(response)
        
        return response.json()
    
    async def list_models(self) -> List[str]:
        """List available models."""
        response = await self._request("GET", "/models")
        return response["models"]
    
    async def create_run(
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
    ) -> AsyncSignalRun:
        """Create a new training run."""
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
        
        response_data = await self._request("POST", "/runs", json=config.model_dump())
        response = RunResponse(**response_data)
        
        return AsyncSignalRun(
            client=self,
            run_id=response.run_id,
            config=response.config,
        )
    
    async def forward_backward(
        self,
        run_id: str,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ForwardBackwardResponse:
        """Perform forward-backward pass."""
        if loss_kwargs is None:
            loss_kwargs = {}
            
        payload = {
            "batch_data": batch,
            "accumulate": accumulate,
            "loss_fn": loss_fn,
            "loss_kwargs": loss_kwargs,
        }
        
        response_data = await self._request("POST", f"/runs/{run_id}/forward_backward", json=payload)
        return ForwardBackwardResponse(**response_data)
    
    async def forward_backward_async(
        self,
        run_id: str,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> APIFuture:
        """Perform forward-backward pass (async)."""
        if loss_kwargs is None:
            loss_kwargs = {}
            
        payload = {
            "batch_data": batch,
            "accumulate": accumulate,
            "loss_fn": loss_fn,
            "loss_kwargs": loss_kwargs,
        }
        
        response = await self._request("POST", f"/runs/{run_id}/forward_backward_async", json=payload)
        return APIFuture(client=self, future_id=response["future_id"])
    
    async def optim_step(
        self,
        run_id: str,
        learning_rate: Optional[float] = None,
    ) -> OptimStepResponse:
        """Apply optimizer step."""
        payload = {
            "learning_rate": learning_rate,
        }
        
        response_data = await self._request("POST", f"/runs/{run_id}/optim_step", json=payload)
        return OptimStepResponse(**response_data)
    
    async def optim_step_async(
        self,
        run_id: str,
        learning_rate: Optional[float] = None,
    ) -> APIFuture:
        """Apply optimizer step (async)."""
        payload = {
            "learning_rate": learning_rate,
        }
        
        response = await self._request("POST", f"/runs/{run_id}/optim_step_async", json=payload)
        return APIFuture(client=self, future_id=response["future_id"])
    
    async def sample(
        self,
        run_id: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> SampleResponse:
        """Generate samples."""
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_logprobs": return_logprobs,
        }
        
        response_data = await self._request("POST", f"/runs/{run_id}/sample", json=payload)
        return SampleResponse(**response_data)
    
    async def sample_async(
        self,
        run_id: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> APIFuture:
        """Generate samples (async)."""
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_logprobs": return_logprobs,
        }
        
        response = await self._request("POST", f"/runs/{run_id}/sample_async", json=payload)
        return APIFuture(client=self, future_id=response["future_id"])
    
    async def save_state(
        self,
        run_id: str,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> SaveStateResponse:
        """Save model state."""
        payload = {
            "mode": mode,
            "push_to_hub": push_to_hub,
            "hub_model_id": hub_model_id,
        }
        
        response_data = await self._request("POST", f"/runs/{run_id}/save_state", json=payload)
        return SaveStateResponse(**response_data)
    
    async def save_state_async(
        self,
        run_id: str,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> APIFuture:
        """Save model state (async)."""
        payload = {
            "mode": mode,
            "push_to_hub": push_to_hub,
            "hub_model_id": hub_model_id,
        }
        
        response = await self._request("POST", f"/runs/{run_id}/save_state_async", json=payload)
        return APIFuture(client=self, future_id=response["future_id"])
    
    async def get_run_status(self, run_id: str) -> RunStatus:
        """Get run status."""
        response_data = await self._request("GET", f"/runs/{run_id}/status")
        return RunStatus(**response_data)
    
    async def get_run_metrics(self, run_id: str) -> RunMetrics:
        """Get run metrics."""
        response_data = await self._request("GET", f"/runs/{run_id}/metrics")
        return RunMetrics(**response_data)
    
    async def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs."""
        response = await self._request("GET", "/runs")
        return response["runs"]
    
    async def tokenize(
        self,
        run_id: str,
        text: str | List[str],
        add_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        """Tokenize text using the model's tokenizer."""
        return await self._request(
            "POST",
            f"/runs/{run_id}/tokenize",
            json={"text": text, "add_special_tokens": add_special_tokens},
        )
    
    async def detokenize(
        self,
        run_id: str,
        token_ids: List[int] | List[List[int]],
    ) -> Dict[str, Any]:
        """Detokenize token IDs using the model's tokenizer."""
        return await self._request(
            "POST",
            f"/runs/{run_id}/detokenize",
            json={"token_ids": token_ids},
        )
    
    async def get_tokenizer_info(self, run_id: str) -> Dict[str, Any]:
        """Get tokenizer configuration information."""
        return await self._request("GET", f"/runs/{run_id}/tokenizer_info")
    
    async def get_model_info(self, run_id: str) -> Dict[str, Any]:
        """Get model architecture information."""
        return await self._request("GET", f"/runs/{run_id}/model_info")
    
    async def apply_chat_template(
        self,
        run_id: str,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> Dict[str, Any]:
        """Apply the model's chat template to format messages."""
        return await self._request(
            "POST",
            f"/runs/{run_id}/apply_chat_template",
            json={"messages": messages, "add_generation_prompt": add_generation_prompt},
        )
