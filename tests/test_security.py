"""Security tests for the Signal API."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app
from api.auth import HybridAuthManager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch('api.main.auth_manager') as mock:
        yield mock


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_invalid_lora_r_rejected(self, client, mock_auth):
        """Test that invalid LoRA rank is rejected."""
        mock_auth.validate_api_key.return_value = "test-user-123"
        
        response = client.post(
            "/runs",
            json={
                "base_model": "meta-llama/Llama-3.2-3B",
                "lora_r": -10  # Invalid negative value
            },
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_learning_rate_rejected(self, client, mock_auth):
        """Test that invalid learning rate is rejected."""
        mock_auth.validate_api_key.return_value = "test-user-123"
        
        response = client.post(
            "/runs",
            json={
                "base_model": "meta-llama/Llama-3.2-3B",
                "learning_rate": 10.0  # Too high
            },
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        assert response.status_code == 422
    
    def test_path_traversal_prevention(self, client, mock_auth):
        """Test that path traversal in model names is prevented."""
        mock_auth.validate_api_key.return_value = "test-user-123"
        
        response = client.post(
            "/runs",
            json={
                "base_model": "../../etc/passwd",  # Path traversal attempt
                "lora_r": 32
            },
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        assert response.status_code == 422
    
    def test_batch_size_limits(self, client, mock_auth):
        """Test that batch size is limited."""
        mock_auth.validate_api_key.return_value = "test-user-123"
        
        # Create a batch with too many examples
        large_batch = [{"text": f"example {i}"} for i in range(200)]  # > 128 limit
        
        response = client.post(
            "/runs/test-run/forward_backward",
            json={"batch_data": large_batch},
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        assert response.status_code == 422


class TestAuthentication:
    """Test authentication mechanisms."""
    
    def test_missing_auth_header(self, client):
        """Test that missing auth header is rejected."""
        response = client.post(
            "/runs",
            json={"base_model": "meta-llama/Llama-3.2-3B", "lora_r": 32}
        )
        
        assert response.status_code == 401
        assert "Missing authorization header" in response.json()["error"]
    
    def test_invalid_auth_format(self, client):
        """Test that invalid auth format is rejected."""
        response = client.post(
            "/runs",
            json={"base_model": "meta-llama/Llama-3.2-3B"},
            headers={"Authorization": "InvalidFormat"}
        )
        
        assert response.status_code == 401


class TestAPIKeySecurity:
    """Test API key security features."""
    
    def test_api_key_bcrypt_hashing(self):
        """Test that API keys are hashed with bcrypt."""
        auth_manager = HybridAuthManager()
        
        key = "sk-test-key-12345"
        hash1 = auth_manager._hash_key(key)
        hash2 = auth_manager._hash_key(key)
        
        # Hashes should be different (due to random salt)
        assert hash1 != hash2
        
        # But both should verify correctly
        assert auth_manager._verify_key_hash(key, hash1)
        assert auth_manager._verify_key_hash(key, hash2)
    
    def test_invalid_key_verification(self):
        """Test that invalid keys are rejected."""
        auth_manager = HybridAuthManager()
        
        key = "sk-test-key-12345"
        wrong_key = "sk-wrong-key-67890"
        hash_value = auth_manager._hash_key(key)
        
        # Wrong key should not verify
        assert not auth_manager._verify_key_hash(wrong_key, hash_value)


class TestErrorSanitization:
    """Test that error messages don't leak sensitive information."""
    
    def test_internal_errors_sanitized(self, client, mock_auth):
        """Test that internal errors don't expose details."""
        mock_auth.validate_api_key.return_value = "test-user-123"
        
        with patch('api.main.run_registry') as mock_registry:
            mock_registry.create_run.side_effect = Exception("Database connection failed with password=secret123")
            
            response = client.post(
                "/runs",
                json={"base_model": "meta-llama/Llama-3.2-3B", "lora_r": 32},
                headers={"Authorization": "Bearer sk-test-key"}
            )
            
            # Should not contain internal error details
            assert response.status_code == 500
            error_data = response.json()
            assert "password" not in str(error_data).lower()
            assert "Internal server error" in error_data["error"]


class TestResourceLimits:
    """Test resource limit enforcement."""
    
    def test_concurrent_run_limit_enforced(self, client, mock_auth):
        """Test that concurrent run limit is enforced."""
        mock_auth.validate_api_key.return_value = "test-user-123"
        
        with patch('api.main.run_registry') as mock_registry:
            # Mock that user already has 5 active runs
            mock_registry.create_run.side_effect = Exception(
                "Maximum 5 concurrent runs reached"
            )
            
            response = client.post(
                "/runs",
                json={"base_model": "meta-llama/Llama-3.2-3B"},
                headers={"Authorization": "Bearer sk-test-key"}
            )
            
            assert response.status_code == 500
            assert "concurrent runs" in str(response.json()).lower()


class TestSecurityHeaders:
    """Test security headers are present."""
    
    def test_security_headers_present(self, client):
        """Test that security headers are set."""
        response = client.get("/health")
        
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
