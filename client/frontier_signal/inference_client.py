"""Inference-specialized client for Signal API."""

import requests
import time
from typing import List, Dict, Any, Optional

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
        """Initialize inference client.
        
        Args:
            run_id: Run identifier
            api_key: API key for authentication
            base_url: Base URL of the API server
            step: Checkpoint step to use (latest if None)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Number of retries for failed requests (default: 5)
            batch_size: Batch size for inference (default: 1)
            session: Optional shared session (for connection pooling)
        """
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
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })
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
        """Make a request with immediate retry.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            json: Optional JSON payload
            
        Returns:
            Response data
            
        Raises:
            SignalAPIError: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    json=json,
                    timeout=self.timeout
                )
                
                if response.status_code >= 400:
                    response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_exception = SignalTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout}s"
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = SignalConnectionError(
                    f"Failed to connect to {url}: {str(e)}"
                )
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
        """Generate text from prompts.
        
        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_logprobs: Whether to return log probabilities
            step: Checkpoint step to use (overrides instance step)
            
        Returns:
            List of generated texts
        """
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
        
        result = self._request(
            "POST",
            f"/runs/{self.run_id}/sample",
            json=payload
        )
        
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
        """Generate text from prompts in batches.
        
        This automatically batches prompts based on self.batch_size.
        
        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            step: Checkpoint step to use (overrides instance step)
            
        Returns:
            List of generated texts (same order as prompts)
        """
        all_outputs = []
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
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
        """Stream tokens as they're generated (future feature).
        
        Args:
            prompt: Single prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            step: Checkpoint step to use
            
        Yields:
            Generated tokens one at a time
            
        Note:
            This is a placeholder for future streaming support.
            Currently falls back to regular sample().
        """
        # TODO: Implement streaming when API supports it
        # For now, just return the full output
        result = self.sample(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            step=step,
        )
        
        if result:
            yield result[0]
    
    def embeddings(
        self,
        texts: List[str],
        step: Optional[int] = None,
    ) -> List[List[float]]:
        """Get embeddings for texts (future feature).
        
        Args:
            texts: List of texts to embed
            step: Checkpoint step to use
            
        Returns:
            List of embedding vectors
            
        Note:
            This is a placeholder for future embedding support.
        """
        # TODO: Implement when API supports embeddings
        raise NotImplementedError(
            "Embeddings are not yet supported. "
            "This feature will be added in a future release."
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache size and hit rate info
        """
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }

