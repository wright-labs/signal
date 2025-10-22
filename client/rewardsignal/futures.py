"""API-based futures for async execution.

This module provides futures that poll the Signal API for results,
enabling true async execution of training operations.
"""

from typing import Any, Dict, Optional, List
import asyncio


class APIFuture:
    """Future that polls the API for async results.

    This polls the /futures/{id} endpoint until the operation completes.
    Unlike Modal futures (which require direct Modal access), this works
    through the HTTP API.

    Examples:
        >>> # Async usage
        >>> future = await client.forward_backward(..., return_future=True)
        >>> result = await future

        >>> # Sync usage
        >>> future = await client.forward_backward(..., return_future=True)
        >>> result = future.result()
    """

    def __init__(self, client, future_id: str):
        """Initialize API future.

        Args:
            client: AsyncSignalClient instance
            future_id: Unique future identifier
        """
        self.client = client
        self.future_id = future_id
        self._result = None
        self._completed = False
        self._error = None

    async def _poll_until_complete(self):
        """Poll API until future completes."""
        while not self._completed:
            response = await self.client._request("GET", f"/futures/{self.future_id}")

            if response["status"] == "completed":
                self._result = response["result"]
                self._completed = True
            elif response["status"] == "failed":
                self._error = response["error"]
                self._completed = True
                raise Exception(self._error)
            else:
                # Still pending, wait before polling again
                await asyncio.sleep(0.1)

        return self._result

    def __await__(self):
        """Await the future."""
        return self._poll_until_complete().__await__()

    def result(self, timeout: Optional[float] = None):
        """Get result synchronously (blocking).

        Args:
            timeout: Maximum seconds to wait (not implemented yet)

        Returns:
            Future result
        """
        return asyncio.run(self._poll_until_complete())

    def cancel(self) -> bool:
        """Cancel the future (attempts to cancel on server).

        Returns:
            True if successfully cancelled, False otherwise
        """
        if self._completed:
            return False

        try:
            # Try to cancel via API
            asyncio.run(self.client._request("DELETE", f"/futures/{self.future_id}"))
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation."""
        status = "completed" if self._completed else "pending"
        return f"<APIFuture(id={self.future_id[:8]}..., status={status})>"


class FutureGroup:
    """Manage multiple API futures as a group.

    Useful for batch operations and waiting on multiple futures.

    Examples:
        >>> group = FutureGroup()
        >>> for batch in batches:
        >>>     future = await client.forward_backward(batch, return_future=True)
        >>>     group.add(future)
        >>> results = await group.wait_all()
    """

    def __init__(self):
        """Initialize empty future group."""
        self.futures: List[APIFuture] = []

    def add(self, future: APIFuture) -> None:
        """Add a future to the group.

        Args:
            future: APIFuture to add
        """
        self.futures.append(future)

    async def wait_all(self) -> List[Dict[str, Any]]:
        """Wait for all futures to complete.

        Returns:
            List of results
        """
        return await asyncio.gather(*[f._poll_until_complete() for f in self.futures])

    def __len__(self) -> int:
        """Get number of futures in group."""
        return len(self.futures)

    def __iter__(self):
        """Iterate over futures."""
        return iter(self.futures)
