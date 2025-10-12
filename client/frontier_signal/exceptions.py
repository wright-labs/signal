"""Custom exceptions for Signal SDK."""

from typing import Optional, Dict, Any


class SignalAPIError(Exception):
    """Base exception for all Signal API errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message
            status_code: HTTP status code if applicable
            response_data: Raw response data from the API
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
    
    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(SignalAPIError):
    """Raised when authentication fails (401)."""
    
    def __init__(self, message: str = "Authentication failed. Check your API key.", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(SignalAPIError):
    """Raised when user is not authorized to access a resource (403)."""
    
    def __init__(self, message: str = "You are not authorized to access this resource.", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class NotFoundError(SignalAPIError):
    """Raised when a resource is not found (404)."""
    
    def __init__(self, message: str = "Resource not found.", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class ValidationError(SignalAPIError):
    """Raised when request validation fails (422)."""
    
    def __init__(self, message: str = "Request validation failed.", **kwargs):
        super().__init__(message, status_code=422, **kwargs)


class RateLimitError(SignalAPIError):
    """Raised when rate limit is exceeded (429)."""
    
    def __init__(self, message: str = "Rate limit exceeded. Please try again later.", **kwargs):
        super().__init__(message, status_code=429, **kwargs)


class ServerError(SignalAPIError):
    """Raised when server encounters an error (5xx)."""
    
    def __init__(self, message: str = "Server error occurred.", **kwargs):
        super().__init__(message, **kwargs)


class ConnectionError(SignalAPIError):
    """Raised when connection to the API fails."""
    
    def __init__(self, message: str = "Failed to connect to Signal API.", **kwargs):
        super().__init__(message, **kwargs)


class TimeoutError(SignalAPIError):
    """Raised when request times out."""
    
    def __init__(self, message: str = "Request timed out.", **kwargs):
        super().__init__(message, **kwargs)
