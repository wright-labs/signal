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
    
    async def get_balance(self, user_id: str) -> float:
        """Get user's current credit balance."""
        if not self._is_configured():
            logger.warning("Frontier client not configured - returning zero balance")
            return 0
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/get-balance",
                    json={"user_id": user_id},
                    headers={"X-Internal-Secret": self.internal_secret}
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
        self, 
        user_id: str, 
        amount: float, 
        run_id: str, 
        step: Optional[int] = None
    ) -> bool:
        """Charge credits immediately for incremental billing with idempotency."""
        if not self._is_configured():
            logger.warning("Frontier client not configured - skipping charge")
            return True
        
        # Generate idempotency key (deterministic based on run_id, step, and amount)
        # Use cents to avoid floating point issues
        amount_cents = int(amount * 10000)
        idempotency_key = f"{run_id}-step{step or 0}-{amount_cents}"
        
        # Check if already charged (deduplication)
        from api.supabase_client import get_supabase
        supabase = get_supabase()
        
        try:
            existing = supabase.table("billing_charges").select("*").eq(
                "idempotency_key", idempotency_key
            ).execute()
            
            if existing.data:
                logger.info(f"Charge deduplicated for {idempotency_key} - already processed")
                return True  # Already charged, return success
        except Exception as e:
            logger.warning(f"Failed to check idempotency: {e} - proceeding with charge")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.backend_url}/internal/charge-credits",
                    json={
                        "user_id": user_id,
                        "amount": amount,
                        "run_id": run_id,
                        "step": step,
                        "description": f"Training run {run_id}" + (f" step {step}" if step else "")
                    },
                    headers={"X-Internal-Secret": self.internal_secret}
                )
                
                if response.status_code == 200:
                    # Record successful charge in database
                    try:
                        supabase.table("billing_charges").insert({
                            "idempotency_key": idempotency_key,
                            "run_id": run_id,
                            "user_id": user_id,
                            "amount": amount,
                            "step": step,
                        }).execute()
                    except Exception as e:
                        logger.warning(f"Failed to record charge: {e} - charge succeeded but not recorded")
                    
                    logger.info(f"Charged ${amount:.4f} for run {run_id}")
                    return True
                elif response.status_code == 402:
                    logger.warning(f"Insufficient credits for run {run_id}")
                    return False
                else:
                    logger.error(f"Charge failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to charge credits: {e}")
            return False
    
    async def get_integrations(self, user_id: str) -> Dict[str, str]:
        """Get user's integration credentials."""
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

