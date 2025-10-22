"""Synchronous Python client SDK for Signal API."""

import requests
from typing import List, Dict, Any, Optional, Literal

from .schemas import (
    RunConfig,
    RunResponse,
    RunStatus,
    RunMetrics,
    ForwardBackwardResponse,
    OptimStepResponse,
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
from .training_client import TrainingClient
from .inference_client import InferenceClient


class SignalRun:
    """Represents a training run with convenient methods."""

    def __init__(self, client: "SignalClient", run_id: str, config: Dict[str, Any]):
        self.client = client
        self.run_id = run_id
        self.config = config

    def forward_backward(
        self,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        **loss_kwargs,
    ) -> ForwardBackwardResponse:
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
    ) -> OptimStepResponse:
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
    ) -> List[str]:
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
    ) -> SaveStateResponse:
        return self.client.save_state(
            run_id=self.run_id,
            mode=mode,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )

    def get_status(self) -> RunStatus:
        return self.client.get_run_status(self.run_id)

    def get_metrics(self) -> RunMetrics:
        return self.client.get_run_metrics(self.run_id)

    def training(self, **kwargs) -> "TrainingClient":
        """get just the training client for the run"""
        return self.client.training(self.run_id, **kwargs)

    def inference(self, **kwargs) -> "InferenceClient":
        """get just the inference client for the run"""
        return self.client.inference(self.run_id, **kwargs)

    def tokenize(
        self,
        text: str | List[str],
        add_special_tokens: bool = True,
    ) -> TokenizeResponse:
        """tokenize something using the tokenizer"""
        return self.client.tokenize(
            run_id=self.run_id,
            text=text,
            add_special_tokens=add_special_tokens,
        )

    def detokenize(
        self,
        token_ids: List[int] | List[List[int]],
    ) -> DetokenizeResponse:
        """detokenize something using the tokenizer"""
        return self.client.detokenize(
            run_id=self.run_id,
            token_ids=token_ids,
        )

    def get_tokenizer_info(self) -> TokenizerInfoResponse:
        return self.client.get_tokenizer_info(self.run_id)

    def get_model_info(self) -> ModelInfoResponse:
        """returns model name and size"""
        return self.client.get_model_info(self.run_id)
    
    # TODO: maybe don't ship the chat template thing
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> ApplyChatTemplateResponse:
        """make a chat template for openai compaitb"""
        return self.client.apply_chat_template(
            run_id=self.run_id,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )


class SignalClient:
    """Synchronous client for Signal API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.frontier-signal.com", # TODO: hardcode the default url, 
        timeout: int = 300,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def __enter__(self):
        """context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """context manager exit."""
        self.close()

    def close(self):
        """Close the session."""
        self.session.close()

    def _handle_error(self, response: requests.Response) -> None:
        """did this to centralize error handling"""
        try:
            error_data = response.json()
            error_msg = error_data.get("detail") or error_data.get("error", response.text)
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code} error"

        status_code = response.status_code

        if status_code == 401:
            raise AuthenticationError(
                error_msg, response_data=error_data if "error_data" in locals() else None
            )
        elif status_code == 403:
            raise AuthorizationError(
                error_msg, response_data=error_data if "error_data" in locals() else None
            )
        elif status_code == 404:
            raise NotFoundError(
                error_msg, response_data=error_data if "error_data" in locals() else None
            )
        elif status_code == 422:
            raise ValidationError(
                error_msg, response_data=error_data if "error_data" in locals() else None
            )
        elif status_code == 429:
            raise RateLimitError(
                error_msg, response_data=error_data if "error_data" in locals() else None
            )
        elif status_code >= 500:
            raise ServerError(
                error_msg,
                status_code=status_code,
                response_data=error_data if "error_data" in locals() else None,
            )
        else:
            raise SignalAPIError(
                error_msg,
                status_code=status_code,
                response_data=error_data if "error_data" in locals() else None,
            )

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """make a request to the API woah"""
        url = f"{self.base_url}{endpoint}"

        # yes these are a lot of overspecified try catches, yes claude suggested this, yes i want to keep it for debugging.
        try:
            response = self.session.request(method, url, json=json, timeout=self.timeout)
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
        """creates a new training run."""
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

        response_data = self._request("POST", "/runs", json=config.model_dump())
        response = RunResponse(**response_data)

        return SignalRun(
            client=self,
            run_id=response.run_id,
            config=response.config,
        )

    def forward_backward(
        self,
        run_id: str,
        batch: List[Dict[str, Any]],
        accumulate: bool = False,
        loss_fn: str = "causal_lm",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ForwardBackwardResponse:
        """perform forward-backward pass using training client"""
        training = self.training(run_id)
        return training.forward_backward(
            batch_data=batch,
            accumulate=accumulate,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        )

    def optim_step(
        self,
        run_id: str,
        learning_rate: Optional[float] = None,
    ) -> OptimStepResponse:
        """apply optimizer step using training client"""
        training = self.training(run_id)
        return training.optim_step(learning_rate=learning_rate)

    def sample(
        self,
        run_id: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
    ) -> List[str]:
        """sample using inference client"""
        # TODO: check that this does this with the trained model so rollouts actually update
        inference = self.inference(run_id)
        return inference.sample(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=return_logprobs,
        )

    def save_state(
        self,
        run_id: str,
        mode: Literal["adapter", "merged"] = "adapter",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> SaveStateResponse:
        """save lora adapters or merged model to R2 using training client"""
        training = self.training(run_id)
        return training.save_checkpoint(
            mode=mode,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )

    def get_run_status(self, run_id: str) -> RunStatus:
        response_data = self._request("GET", f"/runs/{run_id}/status")
        return RunStatus(**response_data)

    def get_run_metrics(self, run_id: str) -> RunMetrics:
        response_data = self._request("GET", f"/runs/{run_id}/metrics")
        return RunMetrics(**response_data)

    def list_runs(self) -> List[Dict[str, Any]]:
        response = self._request("GET", "/runs")
        return response["runs"]

    def tokenize(
        self,
        run_id: str,
        text: str | List[str],
        add_special_tokens: bool = True,
    ) -> TokenizeResponse:
        # TODO: i decided to keep this on the API... should maybe move to training or inference client
        response_data = self._request(
            "POST",
            f"/runs/{run_id}/tokenize",
            json={"text": text, "add_special_tokens": add_special_tokens},
        )
        return TokenizeResponse(**response_data)

    def detokenize(
        self,
        run_id: str,
        token_ids: List[int] | List[List[int]],
    ) -> DetokenizeResponse:
        # TODO: i decided to keep this on the API... should maybe move to training or inference client
        response_data = self._request(
            "POST",
            f"/runs/{run_id}/detokenize",
            json={"token_ids": token_ids},
        )
        return DetokenizeResponse(**response_data)

    def get_tokenizer_info(self, run_id: str) -> TokenizerInfoResponse:
        """get the basic info, vocab size, and special tokens"""
        response_data = self._request("GET", f"/runs/{run_id}/tokenizer_info")
        return TokenizerInfoResponse(**response_data)

    def get_model_info(self, run_id: str) -> ModelInfoResponse:
        """model name and size"""
        response_data = self._request("GET", f"/runs/{run_id}/model_info")
        return ModelInfoResponse(**response_data)

    def apply_chat_template(
        self,
        run_id: str,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> ApplyChatTemplateResponse:
        # TODO: basically the same one as before, do we need to ship this?
        response_data = self._request(
            "POST",
            f"/runs/{run_id}/apply_chat_template",
            json={"messages": messages, "add_generation_prompt": add_generation_prompt},
        )
        return ApplyChatTemplateResponse(**response_data)

    def training(
        self, run_id: str, timeout: int = 3600, max_retries: int = 3, **kwargs
    ) -> TrainingClient:
        """get training client for a run"""
        return TrainingClient(
            run_id=run_id,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
            session=self.session,  # Share session for connection pooling
            **kwargs,
        )

    def inference(
        self,
        run_id: str,
        step: Optional[int] = None,
        timeout: int = 30,
        batch_size: int = 1,
        **kwargs,
    ) -> InferenceClient:
        """get inference client for a run"""
        return InferenceClient(
            run_id=run_id,
            api_key=self.api_key,
            base_url=self.base_url,
            step=step,
            timeout=timeout,
            batch_size=batch_size,
            session=self.session,
            **kwargs,
        )
