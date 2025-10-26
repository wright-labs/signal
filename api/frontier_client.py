"""Client for communicating with Frontier Backend for billing operations."""

import os
import httpx
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class FrontierClient:
    """Client for talking to Frontier Backend internal API."""

    def __init__(self):
        self.backend_url = os.getenv("FRONTIER_BACKEND_URL")
        self.internal_secret = os.getenv("SIGNAL_INTERNAL_SECRET")

        if not self.backend_url:
            logger.error(
                "FRONTIER_BACKEND_URL not set - billing service unavailable"
            )
        if not self.internal_secret:
            logger.error(
                "SIGNAL_INTERNAL_SECRET not set - billing service unavailable"
            )

    def _is_configured(self) -> bool:
        """Check if client is properly configured."""
        return bool(self.backend_url and self.internal_secret)
    
    def _raise_if_not_configured(self) -> None:
        """Raise 503 if client is not configured (fail-closed to prevent unpaid usage)."""
        if not self._is_configured():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="Billing service unavailable. Please contact support."
            )

    async def get_balance(self, user_id: str) -> float:
        """Get user's current credit balance."""
        self._raise_if_not_configured()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/get-balance",
                    json={"user_id": user_id},
                    headers={"X-Internal-Secret": self.internal_secret},
                )

                if response.status_code == 200:
                    data = response.json()
                    return float(data["balance"])
                else:
                    logger.error(f"Get balance failed: {response.status_code}")
                    return 0.0

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def charge_increment(
        self, user_id: str, amount: float, run_id: str, step: Optional[int] = None
    ) -> bool:
        """Charge credits immediately for incremental billing with idempotency."""
        self._raise_if_not_configured()

        # Generate idempotency key (deterministic based on run_id, step, and amount)
        # Use cents to avoid floating point issues
        amount_cents = int(amount * 10000)
        idempotency_key = f"{run_id}-step{step or 0}-{amount_cents}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/charge-credits",
                    json={
                        "user_id": user_id,
                        "amount": amount,
                        "run_id": run_id,
                        "step": step,
                        "idempotency_key": idempotency_key,
                        "description": f"Training run {run_id}"
                        + (f" step {step}" if step else ""),
                    },
                    headers={"X-Internal-Secret": self.internal_secret},
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("already_charged"):
                        logger.info(
                            f"Charge deduplicated for {idempotency_key} - already processed"
                        )
                    else:
                        logger.info(f"Charged ${amount:.4f} for run {run_id}")
                    return data.get("success", True)
                else:
                    logger.error(
                        f"Charge failed: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to charge credits: {e}")
            return False

    async def get_integrations(self, user_id: str) -> Dict[str, str]:
        """Get user's integration credentials."""
        self._raise_if_not_configured()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/get-integrations",
                    json={"user_id": user_id},
                    headers={"X-Internal-Secret": self.internal_secret},
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["integrations"]
                else:
                    logger.error(
                        f"Failed to fetch integrations: {response.status_code}"
                    )
                    return {}

        except Exception as e:
            logger.error(f"Failed to fetch integrations: {e}")
            return {}


# Global instance
_frontier_client = None


def get_frontier_client() -> FrontierClient:
    """Get or create Frontier client singleton."""
    global _frontier_client
    if _frontier_client is None:
        _frontier_client = FrontierClient()
    return _frontier_client
