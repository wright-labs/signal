"""API key authentication for Signal API via Frontier Backend.

This module delegates all authentication to the Frontier Backend service.
For self-hosting, you can disable auth by not setting FRONTIER_BACKEND_URL.
"""

import logging
from typing import Optional
from fastapi import HTTPException, Header, Request
from api.frontier_client import get_frontier_client
from api.logging_config import security_logger

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    return request.client.host if request.client else "unknown"


async def verify_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> str:
    """Verify API key and return user_id."""
    ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")
    
    if not authorization:
        security_logger.log_auth_failure(ip, user_agent, "Missing authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        security_logger.log_auth_failure(ip, user_agent, "Invalid authorization format")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    api_key = parts[1]
    
    # Validate via Frontier Backend
    frontier = get_frontier_client()
    user_id = await frontier.validate_api_key(api_key)
    
    if not user_id:
        security_logger.log_auth_failure(ip, user_agent, "Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Log successful auth
    security_logger.log_auth_success(user_id, ip, user_agent)
    
    return user_id
