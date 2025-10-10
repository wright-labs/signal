"""API key authentication for Signal API.

This module handles API key validation only. API key generation is handled
by the Frontier Backend service.

For self-hosting: You can implement your own authentication by modifying this
module or by setting API keys directly in the database.
"""
import os
import bcrypt
import logging
from typing import Optional
from datetime import datetime
from fastapi import HTTPException, Header, Request
from supabase import Client
from api.supabase_client import get_supabase, set_user_context
from api.logging_config import security_logger

logger = logging.getLogger(__name__)


class AuthManager:
    """Manages API key authentication via Supabase.
    
    This is a simplified auth manager that only validates API keys.
    API key generation is handled by the Frontier Backend service.
    """
    
    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the auth manager.
        
        Args:
            supabase_client: Optional Supabase client (for testing)
        """
        self.supabase = supabase_client or get_supabase()
    
    def _verify_key_hash(self, api_key: str, stored_hash: str) -> bool:
        """Verify an API key against its stored hash.
        
        Args:
            api_key: Plain text API key
            stored_hash: Stored bcrypt hash
            
        Returns:
            True if key matches hash, False otherwise
        """
        try:
            return bcrypt.checkpw(api_key.encode('utf-8'), stored_hash.encode('utf-8'))
        except Exception:
            return False
    
    async def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user_id.
        
        Uses a hash prefix index to avoid O(n) table scans.
        The key_prefix column stores first 8 chars of key for quick lookup.
        
        Note: This operation happens BEFORE user context is set.
        The RLS policy allows reading api_keys during validation.
        
        Args:
            api_key: API key (sk-xxx format)
            
        Returns:
            user_id (UUID) if valid, None otherwise
        """
        if not api_key or not api_key.startswith("sk-"):
            return None
        
        try:
            # Extract prefix for indexed lookup (first 11 chars: "sk-" + 8 chars)
            key_prefix = api_key[:11] if len(api_key) >= 11 else api_key
            
            # Query only keys with matching prefix (indexed column)
            # RLS policy allows this SELECT during validation
            result = self.supabase.table("api_keys").select(
                "id, user_id, key_hash, is_active"
            ).eq("is_active", True).eq("key_prefix", key_prefix).execute()
            
            if result.data:
                for key_data in result.data:
                    if self._verify_key_hash(api_key, key_data["key_hash"]):
                        user_id = key_data["user_id"]
                        
                        # Update last_used timestamp
                        try:
                            self.supabase.table("api_keys").update({
                                "last_used": datetime.utcnow().isoformat()
                            }).eq("id", key_data["id"]).execute()
                        except Exception as e:
                            # Don't fail validation if timestamp update fails
                            logger.debug(f"Failed to update last_used: {e}")
                        
                        return user_id
        except Exception as e:
            # Log error server-side but return None (invalid key)
            logger.error(f"API key validation error: {e}")
            return None
        
        return None


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    return request.client.host if request.client else "unknown"


async def verify_auth(
    authorization: Optional[str] = Header(None),
    request: Request = None
) -> str:
    """Verify API key authentication and return user_id.
    
    This function:
    1. Validates the API key against the database
    2. Sets the user context for RLS policies
    3. Returns the user_id for use in endpoints
    
    Only supports API key authentication (sk-xxx format).
    API keys are generated and managed by the Frontier Backend service.
    
    For self-hosting: Set up API keys directly in your Supabase database
    or implement custom authentication logic in this function.
    """
    auth_manager = AuthManager()
    
    if not authorization:
        # Log auth failure
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Missing authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    api_key = parts[1]
    
    # Validate API key
    user_id = await auth_manager.validate_api_key(api_key)
    if not user_id:
        # Log auth failure
        if request:
            ip = get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            security_logger.log_auth_failure(ip, user_agent, "Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Set user context for RLS policies
    # This allows all subsequent database queries to be automatically filtered
    if not set_user_context(user_id):
        logger.error(f"Failed to set user context for user {user_id}")
        raise HTTPException(
            status_code=500, 
            detail="Authentication system error"
        )
    
    return user_id
