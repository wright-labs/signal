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
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables"
        )
    
    return create_client(supabase_url, supabase_key)
