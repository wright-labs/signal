"""Inference-specialized client for Signal API."""

import requests
import time
from typing import List, Dict, Any, Optional

from .schemas import (
    TokenizeResponse,
    DetokenizeResponse,
    TokenizerInfoResponse,
    ModelInfoResponse,
    ApplyChatTemplateResponse,
)
from .exceptions import (
    SignalAPIError,
    ConnectionError as SignalConnectionError,
    TimeoutError as SignalTimeoutError,
)


class InferenceClient:
    """Specialized client for inference operations with optimized defaults."""

    def __init__(
        self,
        run_id: str,
        api_key: str,
        base_url: str = "https://api.frontier-signal.com",
        step: Optional[int] = None,
        timeout: int = 30,  # Fast timeout for inference
        max_retries: int = 5,  # More retries for transient failures
        batch_size: int = 1,
        session: Optional[requests.Session] = None,
    ):
        """Initialize inference client."""
        self.run_id = run_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.step = step
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

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

        # Simple cache for repeated prompts
        self._cache: Dict[str, str] = {}
        self._cache_enabled = False

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

    def enable_cache(self):
        """Enable caching for repeated prompts."""
        self._cache_enabled = True

    def disable_cache(self):
        """Disable caching and clear cache."""
        self._cache_enabled = False
        self._cache.clear()

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a request with immediate retry."""
        url = f"{self.base_url}{endpoint}"
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, json=json, timeout=self.timeout)

                if response.status_code >= 400:
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

            # Immediate retry (no backoff for inference)
            if attempt < self.max_retries - 1:
                time.sleep(0.1)  # Brief pause

        # All retries failed
        raise last_exception

    def sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
        step: Optional[int] = None,
    ) -> List[str]:
        """Generate text from prompts."""
        # Check cache if enabled
        if self._cache_enabled and len(prompts) == 1:
            cache_key = f"{prompts[0]}:{max_tokens}:{temperature}:{top_p}:{step}"
            if cache_key in self._cache:
                return [self._cache[cache_key]]

        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_logprobs": return_logprobs,
        }

        # Override step if provided
        if step is not None:
            payload["step"] = step
        elif self.step is not None:
            payload["step"] = self.step

        result = self._request("POST", f"/runs/{self.run_id}/sample", json=payload)

        outputs = result.get("outputs", [])

        # Cache result if enabled and single prompt
        if self._cache_enabled and len(prompts) == 1 and outputs:
            cache_key = f"{prompts[0]}:{max_tokens}:{temperature}:{top_p}:{step}"
            self._cache[cache_key] = outputs[0]

        return outputs

    def batch_sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        step: Optional[int] = None,
    ) -> List[str]:
        """Generate text from prompts in batches."""
        all_outputs = []

        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_outputs = self.sample(
                prompts=batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                step=step,
            )
            all_outputs.extend(batch_outputs)

        return all_outputs

    def stream_sample(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        step: Optional[int] = None,
    ):
        """Stream tokens as they're generated using Server-Sent Events."""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Override step if provided
        if step is not None:
            payload["step"] = step
        elif self.step is not None:
            payload["step"] = self.step

        url = f"{self.base_url}/runs/{self.run_id}/sample/stream"

        try:
            response = self.session.post(
                url,
                json=payload,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Parse SSE events
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        try:
                            import json

                            chunk_data = json.loads(data_str)
                            yield chunk_data

                            # Stop if generation is finished
                            if chunk_data.get("is_finished"):
                                break
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            raise SignalAPIError(f"Streaming request failed: {str(e)}")

    def embeddings(
        self,
        texts: List[str],
        step: Optional[int] = None,
        layer: int = -1,
        pooling: str = "mean",
    ) -> List[List[float]]:
        """Get embeddings for texts."""
        payload = {
            "texts": texts,
            "layer": layer,
            "pooling": pooling,
        }

        # Override step if provided
        if step is not None:
            payload["step"] = step
        elif self.step is not None:
            payload["step"] = self.step

        result = self._request("POST", f"/runs/{self.run_id}/embeddings", json=payload)

        return result.get("embeddings", [])

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }

    def tokenize(
        self,
        text: str | List[str],
        add_special_tokens: bool = True,
    ) -> TokenizeResponse:
        """Tokenize text using the model's tokenizer."""
        response_data = self._request(
            "POST",
            f"/runs/{self.run_id}/tokenize",
            json={"text": text, "add_special_tokens": add_special_tokens},
        )
        return TokenizeResponse(**response_data)

    def detokenize(
        self,
        token_ids: List[int] | List[List[int]],
    ) -> DetokenizeResponse:
        """Detokenize token IDs using the model's tokenizer."""
        response_data = self._request(
            "POST",
            f"/runs/{self.run_id}/detokenize",
            json={"token_ids": token_ids},
        )
        return DetokenizeResponse(**response_data)

    def get_tokenizer_info(self) -> TokenizerInfoResponse:
        """Get tokenizer configuration information."""
        response_data = self._request("GET", f"/runs/{self.run_id}/tokenizer_info")
        return TokenizerInfoResponse(**response_data)

    def get_model_info(self) -> ModelInfoResponse:
        """Get model architecture information."""
        response_data = self._request("GET", f"/runs/{self.run_id}/model_info")
        return ModelInfoResponse(**response_data)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> ApplyChatTemplateResponse:
        """Apply the model's chat template to format messages."""
        response_data = self._request(
            "POST",
            f"/runs/{self.run_id}/apply_chat_template",
            json={"messages": messages, "add_generation_prompt": add_generation_prompt},
        )
        return ApplyChatTemplateResponse(**response_data)
