"""Client for communicating with Frontier Backend for billing operations."""
import os
import httpx
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class FrontierClient:
    """Client for Frontier Backend internal API."""
    
    def __init__(self):
        self.backend_url = os.getenv("FRONTIER_BACKEND_URL")
        self.internal_secret = os.getenv("SIGNAL_INTERNAL_SECRET")
        
        if not self.backend_url:
            logger.warning("FRONTIER_BACKEND_URL not set - credit operations will be disabled")
        if not self.internal_secret:
            logger.warning("SIGNAL_INTERNAL_SECRET not set - credit operations will be disabled")
    
    def _is_configured(self) -> bool:
        """Check if client is properly configured."""
        return bool(self.backend_url and self.internal_secret)
    
    async def validate_credits(self, user_id: str, estimated_cost: float) -> bool:
        """Check if user has sufficient credits.
        
        Args:
            user_id: User UUID
            estimated_cost: Estimated cost in USD
            
        Returns:
            True if user has sufficient credits, False otherwise
        """
        if not self._is_configured():
            logger.warning("Frontier client not configured - skipping credit validation")
            return True  # Allow for self-hosting without Frontier Backend
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/validate-credits",
                    json={"user_id": user_id, "estimated_cost": estimated_cost},
                    headers={"X-Internal-Secret": self.internal_secret}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["has_credits"]
                else:
                    logger.error(f"Credit validation failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to validate credits: {e}")
            return False
    
    async def deduct_credits(self, user_id: str, amount: float, run_id: str, description: Optional[str] = None):
        """Deduct credits after job completion.
        
        Args:
            user_id: User UUID
            amount: Amount to deduct in USD
            run_id: Training run ID
            description: Optional description
        """
        if not self._is_configured():
            logger.warning("Frontier client not configured - skipping credit deduction")
            return
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/deduct-credits",
                    json={
                        "user_id": user_id,
                        "amount": amount,
                        "run_id": run_id,
                        "description": description
                    },
                    headers={"X-Internal-Secret": self.internal_secret}
                )
                
                if response.status_code != 200:
                    logger.error(f"Credit deduction failed: {response.status_code} - {response.text}")
                else:
                    logger.info(f"Credits deducted successfully for run {run_id}: ${amount:.4f}")
                    
        except Exception as e:
            logger.error(f"Failed to deduct credits: {e}")
            # Don't raise - we don't want to fail the training run if billing fails
    
    async def get_integrations(self, user_id: str) -> Dict[str, str]:
        """Get user's integration credentials.
        
        Args:
            user_id: User UUID
            
        Returns:
            Dictionary mapping integration type to decrypted API key
        """
        if not self._is_configured():
            logger.warning("Frontier client not configured - returning empty integrations")
            return {}
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/get-integrations",
                    json={"user_id": user_id},
                    headers={"X-Internal-Secret": self.internal_secret}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["integrations"]
                else:
                    logger.error(f"Failed to fetch integrations: {response.status_code}")
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

