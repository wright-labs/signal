"""Logging configuration for security and audit events."""
import os
import logging
import json
from datetime import datetime


class SecurityLogger:
    """Structured logger for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger("security")
        self.logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for production (if enabled)
        if os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true":
            try:
                os.makedirs("logs", exist_ok=True)
                file_handler = logging.FileHandler("logs/security.log")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception:
                # If can't create file handler, just use console
                pass
    
    def log_auth_failure(self, ip: str, user_agent: str, reason: str):
        """Log failed authentication attempts."""
        log_data = {
            "event_type": "auth_failure",
            "ip": ip,
            "user_agent": user_agent,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.warning(f"Authentication failure: {json.dumps(log_data)}")
    
    def log_auth_success(self, user_id: str, ip: str, user_agent: str):
        """Log successful authentication attempts."""
        log_data = {
            "event_type": "auth_success",
            "user_id": user_id,
            "ip": ip,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(f"Authentication success: {json.dumps(log_data)}")
    
    def log_api_key_created(self, user_id: str, ip: str):
        """Log API key creation."""
        log_data = {
            "event_type": "api_key_created",
            "user_id": user_id,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(f"API key created: {json.dumps(log_data)}")
    
    def log_api_key_revoked(self, user_id: str, key_id: str, ip: str):
        """Log API key revocation."""
        log_data = {
            "event_type": "api_key_revoked",
            "user_id": user_id,
            "key_id": key_id,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(f"API key revoked: {json.dumps(log_data)}")
    
    def log_run_created(self, user_id: str, run_id: str, model: str, ip: str):
        """Log expensive run creation."""
        log_data = {
            "event_type": "run_created",
            "user_id": user_id,
            "run_id": run_id,
            "model": model,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(f"Training run created: {json.dumps(log_data)}")
    
    def log_unauthorized_access(self, user_id: str, resource: str, ip: str):
        """Log unauthorized access attempts."""
        log_data = {
            "event_type": "unauthorized_access",
            "user_id": user_id,
            "resource": resource,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.warning(f"Unauthorized access attempt: {json.dumps(log_data)}")
    
    def log_rate_limit_exceeded(self, user_id: str, endpoint: str, ip: str):
        """Log rate limit violations."""
        log_data = {
            "event_type": "rate_limit_exceeded",
            "user_id": user_id,
            "endpoint": endpoint,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.warning(f"Rate limit exceeded: {json.dumps(log_data)}")


# Global security logger instance
security_logger = SecurityLogger()

