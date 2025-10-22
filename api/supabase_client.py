"""Supabase client and utilities."""

import os
import logging
from supabase import create_client, Client
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache()
def get_supabase() -> Client:
    """Get or create Supabase client instance."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            f"SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables. "
            f"URL: {supabase_url}, Key: {supabase_key[:20] if supabase_key else 'None'}..."
        )

    return create_client(supabase_url, supabase_key)


def set_user_context(user_id: str) -> bool:
    """Set user context for RLS policies."""
    try:
        supabase = get_supabase()
        supabase.rpc("set_user_context", {"user_id_param": user_id}).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to set user context: {e}")
        return False
