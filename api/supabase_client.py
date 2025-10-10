"""Supabase client and utilities."""
import os
import logging
from supabase import create_client, Client
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache()
def get_supabase() -> Client:
    """Get or create Supabase client instance.
    
    TEMPORARY: Using service role for testing to bypass RLS.
    TODO: Revert to ANON key after applying RLS functions.
    
    Returns:
        Supabase client instance
        
    Raises:
        ValueError: If required environment variables are not set
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # TEMPORARY: Using service role for testing
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables"
        )
    
    return create_client(supabase_url, supabase_key)


@lru_cache()
def get_supabase_admin() -> Client:
    """Get Supabase client with service role for admin operations.
    
    Only use this for administrative tasks like:
    - Creating API keys (key management scripts)
    - System maintenance operations
    
    WARNING: Service role bypasses RLS. Use sparingly and never in
    user-facing API endpoints.
    
    Returns:
        Supabase client with service role access
        
    Raises:
        ValueError: If required environment variables are not set
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for admin operations"
        )
    
    return create_client(supabase_url, supabase_key)


def set_user_context(user_id: str) -> bool:
    """Set user context for RLS policies.
    
    This must be called after API key validation to establish the user's
    identity for all subsequent database operations. RLS policies will
    automatically filter queries based on this context.
    
    Args:
        user_id: User UUID to set as current context
        
    Returns:
        True if context was set successfully, False otherwise
    """
    try:
        supabase = get_supabase()
        supabase.rpc('set_user_context', {'user_id_param': user_id}).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to set user context: {e}")
        return False

