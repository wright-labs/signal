"""Async inference-specialized client for Signal API."""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional

from .exceptions import (
    SignalAPIError,
    ConnectionError as SignalConnectionError,
    TimeoutError as SignalTimeoutError,
)


class AsyncInferenceClient:
    """Async specialized client for inference operations with optimized defaults."""
    
    def __init__(
        self,
        run_id: str,
        api_key: str,
        base_url: str = "https://api.frontier-signal.com",
        step: Optional[int] = None,
        timeout: int = 30,  # Fast timeout for inference
        max_retries: int = 5,
        batch_size: int = 1,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """Initialize async inference client.
        
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
        
        # Use shared session if provided
        if session is not None:
            self.session = session
            self._owns_session = False
        else:
            self.session = None  # Will be created on first use
            self._owns_session = True
        
        # Simple cache
        self._cache: Dict[str, str] = {}
        self._cache_enabled = False
    
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
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an async request with immediate retry.
        
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
            # Create session if not exists
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
            
            # Immediate retry (brief pause for inference)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.1)
        
        # All retries failed
        raise last_exception
    
    async def sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = False,
        step: Optional[int] = None,
    ) -> List[str]:
        """Generate text from prompts (async).
        
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
        
        result = await self._request(
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
    
    async def batch_sample(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        step: Optional[int] = None,
    ) -> List[str]:
        """Generate text from prompts in batches (async).
        
        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            step: Checkpoint step to use
            
        Returns:
            List of generated texts
        """
        all_outputs = []
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_outputs = await self.sample(
                prompts=batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                step=step,
            )
            all_outputs.extend(batch_outputs)
        
        return all_outputs
    
    async def stream_sample(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        step: Optional[int] = None,
    ):
        """Stream tokens as they're generated (future async feature).
        
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
        """
        # TODO: Implement streaming when API supports it
        result = await self.sample(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            step=step,
        )
        
        if result:
            yield result[0]
    
    async def embeddings(
        self,
        texts: List[str],
        step: Optional[int] = None,
    ) -> List[List[float]]:
        """Get embeddings for texts (future async feature).
        
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
        """Get cache statistics (sync).
        
        Returns:
            Dict with cache size and hit rate info
        """
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }

