"""API key authentication for Signal API.

This module handles API key validation only. API key generation is handled
by the Frontier Backend service.

For self-hosting: Just don't add auth duh.
"""

import os
import bcrypt
import logging
import asyncio
import time
import jwt
from typing import Optional
from datetime import datetime
from fastapi import HTTPException, Header, Request
from supabase import Client
from api.supabase_client import get_supabase, set_user_context
from api.logging_config import security_logger

logger = logging.getLogger(__name__)


class AuthManager:
    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the auth manager."""
        self.supabase = supabase_client or get_supabase()

    def _verify_key_hash(self, api_key: str, stored_hash: str) -> bool:
        """Verify an API key against its stored hash."""
        try:
            return bcrypt.checkpw(api_key.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False

    async def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user_id."""
        start_time = time.time()

        # Always perform validation logic to prevent timing attacks
        is_valid_format = api_key and api_key.startswith("sk-") and len(api_key) >= 11

        try:
            if is_valid_format:
                # Extract first 11 chars as prefix for indexed lookup
                # Format: "sk-" (3 chars) + first 8 chars of token = 11 chars total
                key_prefix = api_key[:11]

                # Query keys with matching prefix (indexed column)
                # Include expires_at for expiration check
                result = (
                    self.supabase.table("api_keys")
                    .select("id, user_id, key_hash, is_active, expires_at")
                    .eq("is_active", True)
                    .eq("key_prefix", key_prefix)
                    .execute()
                )

                if result.data:
                    for key_data in result.data:
                        # Check expiration
                        if key_data.get("expires_at"):
                            expires_at = datetime.fromisoformat(
                                key_data["expires_at"].replace("Z", "+00:00")
                            )
                            if datetime.now(expires_at.tzinfo) >= expires_at:
                                logger.info(
                                    f"Expired API key attempted: {key_data['id']}"
                                )
                                continue

                        # Verify hash
                        if self._verify_key_hash(api_key, key_data["key_hash"]):
                            user_id = key_data["user_id"]

                        # Verify user still exists
                        try:
                            user_check = (
                                self.supabase.table("profiles")
                                .select("id")
                                .eq("id", user_id)
                                .execute()
                            )
                            if not user_check.data:
                                logger.warning(f"API key for deleted user: {user_id}")
                                await self._normalize_timing(start_time)
                                return None
                        except (ConnectionError, TimeoutError) as e:
                            # Network errors are acceptable - fail closed
                            logger.error(
                                f"Database connection failed during user verification: {e}"
                            )
                            await self._normalize_timing(start_time)
                            return None
                        except Exception as e:
                            # Unexpected errors should be investigated
                            logger.exception(
                                f"Unexpected error verifying user existence: {e}"
                            )
                            await self._normalize_timing(start_time)
                            return None

                        # Update last_used timestamp
                        try:
                            self.supabase.table("api_keys").update(
                                {"last_used": datetime.utcnow().isoformat()}
                            ).eq("id", key_data["id"]).execute()
                        except (ConnectionError, TimeoutError):
                            # Non-critical update, acceptable to skip
                            logger.debug(
                                "Skipped last_used update due to connection issue"
                            )
                        except Exception as e:
                            # Unexpected errors should be logged with full context
                            logger.exception(
                                f"Unexpected error updating last_used for key {key_data['id']}: {e}"
                            )

                            await self._normalize_timing(start_time, target_ms=50)
                            return user_id
        except Exception as e:
            logger.error(f"API key validation error: {e}")

        await self._normalize_timing(start_time, target_ms=50)
        return None

    async def _normalize_timing(self, start_time: float, target_ms: float = 50):
        """Add delay to normalize response timing."""
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms < target_ms:
            delay_ms = target_ms - elapsed_ms
            await asyncio.sleep(delay_ms / 1000)

    async def validate_jwt_token(self, token: str) -> Optional[str]:
        """Validate JWT token from Frontier Backend."""
        start_time = time.time()

        try:
            # Get JWT configuration from environment
            jwt_secret = os.getenv("FRONTIER_JWT_SECRET")
            jwt_algorithm = os.getenv("FRONTIER_JWT_ALGORITHM", "HS256")

            if not jwt_secret:
                logger.warning(
                    "FRONTIER_JWT_SECRET not configured, JWT validation disabled"
                )
                await self._normalize_timing(start_time)
                return None

            # Decode and validate JWT
            payload = jwt.decode(
                token,
                jwt_secret,
                algorithms=[jwt_algorithm],
                options={
                    "require_exp": True,  # Require expiration
                    "require_iat": True,  # Require issued-at
                    "verify_exp": True,  # Verify not expired
                    "verify_iat": True,  # Verify issued-at is valid
                },
            )

            # Extract user_id from standard 'sub' (subject) claim
            user_id = payload.get("sub")
            if not user_id:
                logger.warning("JWT missing 'sub' claim")
                await self._normalize_timing(start_time)
                return None

            # Verify user still exists in database
            try:
                user_check = (
                    self.supabase.table("profiles")
                    .select("id")
                    .eq("id", user_id)
                    .execute()
                )
                if not user_check.data:
                    logger.warning(f"JWT for deleted user: {user_id}")
                    await self._normalize_timing(start_time)
                    return None
            except (ConnectionError, TimeoutError) as e:
                logger.error(
                    f"Database connection failed during JWT user verification: {e}"
                )
                await self._normalize_timing(start_time)
                return None
            except Exception as e:
                logger.exception(f"Unexpected error verifying JWT user existence: {e}")
                await self._normalize_timing(start_time)
                return None

            await self._normalize_timing(start_time, target_ms=50)
            return user_id

        except jwt.ExpiredSignatureError:
            logger.info("JWT token expired")
        except jwt.InvalidTokenError as e:
            logger.info(f"Invalid JWT token: {e}")
        except Exception as e:
            logger.error(f"JWT validation error: {e}")

        await self._normalize_timing(start_time, target_ms=50)
        return None


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    return request.client.host if request.client else "unknown"


async def verify_auth(
    authorization: Optional[str] = Header(None), request: Optional[Request] = None
) -> str:
    """Verify API key authentication and return user_id"""
    auth_manager = AuthManager()
    ip = get_client_ip(request) if request else "unknown"
    user_agent = request.headers.get("user-agent", "unknown") if request else "unknown"

    if not authorization:
        # Log auth failure
        security_logger.log_auth_failure(ip, user_agent, "Missing authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization header")

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        security_logger.log_auth_failure(
            ip, user_agent, "Invalid authorization header format"
        )
        raise HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    api_key = parts[1]

    # Validate API key
    user_id = await auth_manager.validate_api_key(api_key)
    if not user_id:
        # Log auth failure
        security_logger.log_auth_failure(ip, user_agent, "Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Set user context for RLS policies
    # This allows all subsequent database queries to be automatically filtered
    if not set_user_context(user_id):
        logger.error(f"Failed to set user context for user {user_id}")
        raise HTTPException(status_code=500, detail="Authentication system error")

    # Log successful authentication
    security_logger.log_auth_success(user_id, ip, user_agent)

    return user_id
