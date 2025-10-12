"""Tests for authentication and API key management."""
import pytest
from api.auth import APIKeyManager


class TestAPIKeyGeneration:
    """Tests for API key generation."""
    
    def test_generate_key_format(self, api_key_manager, test_user_id):
        """Test that generated keys have correct format."""
        api_key = api_key_manager.generate_key(test_user_id, "Test key")
        
        assert api_key.startswith("sk-")
        assert len(api_key) > 10
    
    def test_generate_key_stores_hash(self, api_key_manager, test_user_id):
        """Test that keys are stored as hashes, not plaintext."""
        api_key = api_key_manager.generate_key(test_user_id, "Test key")
        
        # Load the keys file and verify the key is not stored in plaintext
        data = api_key_manager._load_keys()
        
        # The plaintext key should not appear in the file
        assert api_key not in str(data)
        
        # But we should be able to validate it
        assert api_key_manager.validate_key(api_key) == test_user_id
    
    def test_generate_key_with_description(self, api_key_manager, test_user_id):
        """Test generating key with description."""
        description = "My test API key"
        api_key = api_key_manager.generate_key(test_user_id, description)
        
        keys = api_key_manager.list_keys(test_user_id)
        assert len(keys) == 1
        assert keys[0]["description"] == description
    
    def test_generate_multiple_keys(self, api_key_manager, test_user_id):
        """Test generating multiple keys for same user."""
        key1 = api_key_manager.generate_key(test_user_id, "Key 1")
        key2 = api_key_manager.generate_key(test_user_id, "Key 2")
        
        assert key1 != key2
        assert api_key_manager.validate_key(key1) == test_user_id
        assert api_key_manager.validate_key(key2) == test_user_id
        
        keys = api_key_manager.list_keys(test_user_id)
        assert len(keys) == 2


class TestAPIKeyValidation:
    """Tests for API key validation."""
    
    def test_validate_correct_key(self, api_key_manager, test_api_key, test_user_id):
        """Test validating a correct API key."""
        user_id = api_key_manager.validate_key(test_api_key)
        assert user_id == test_user_id
    
    def test_validate_invalid_key(self, api_key_manager):
        """Test validating an invalid API key."""
        user_id = api_key_manager.validate_key("sk-invalid-key-12345")
        assert user_id is None
    
    def test_validate_malformed_key(self, api_key_manager):
        """Test validating keys without sk- prefix."""
        assert api_key_manager.validate_key("invalid-format") is None
        assert api_key_manager.validate_key("") is None
        assert api_key_manager.validate_key("Bearer sk-test") is None
    
    def test_validate_none_key(self, api_key_manager):
        """Test validating None as key."""
        assert api_key_manager.validate_key(None) is None
    
    def test_validate_updates_last_used(self, api_key_manager, test_api_key, test_user_id):
        """Test that validation updates last_used timestamp."""
        # First validation
        api_key_manager.validate_key(test_api_key)
        
        keys = api_key_manager.list_keys(test_user_id)
        assert keys[0]["last_used"] is not None
        
        first_used = keys[0]["last_used"]
        
        # Second validation
        import time
        time.sleep(0.01)  # Small delay to ensure timestamp changes
        api_key_manager.validate_key(test_api_key)
        
        keys = api_key_manager.list_keys(test_user_id)
        assert keys[0]["last_used"] != first_used


class TestKeyManagement:
    """Tests for key management operations."""
    
    def test_list_keys_for_user(self, api_key_manager, test_user_id):
        """Test listing keys for a user."""
        # Generate multiple keys
        api_key_manager.generate_key(test_user_id, "Key 1")
        api_key_manager.generate_key(test_user_id, "Key 2")
        
        keys = api_key_manager.list_keys(test_user_id)
        assert len(keys) == 2
        
        # Verify structure
        for key_info in keys:
            assert "user_id" in key_info
            assert "description" in key_info
            assert "created_at" in key_info
            assert "last_used" in key_info
            assert key_info["user_id"] == test_user_id
    
    def test_list_keys_empty(self, api_key_manager):
        """Test listing keys for user with no keys."""
        keys = api_key_manager.list_keys("nonexistent_user")
        assert keys == []
    
    def test_revoke_key(self, api_key_manager, test_api_key, test_user_id):
        """Test revoking an API key."""
        # Verify key works before revocation
        assert api_key_manager.validate_key(test_api_key) == test_user_id
        
        # Revoke the key
        result = api_key_manager.revoke_key(test_api_key)
        assert result is True
        
        # Verify key no longer works
        assert api_key_manager.validate_key(test_api_key) is None
        
        # Verify key is not in list
        keys = api_key_manager.list_keys(test_user_id)
        assert len(keys) == 0
    
    def test_revoke_nonexistent_key(self, api_key_manager):
        """Test revoking a key that doesn't exist."""
        result = api_key_manager.revoke_key("sk-nonexistent-key")
        assert result is False
    
    def test_revoke_does_not_affect_other_keys(self, api_key_manager, test_user_id):
        """Test that revoking one key doesn't affect others."""
        key1 = api_key_manager.generate_key(test_user_id, "Key 1")
        key2 = api_key_manager.generate_key(test_user_id, "Key 2")
        
        # Revoke key1
        api_key_manager.revoke_key(key1)
        
        # key2 should still work
        assert api_key_manager.validate_key(key2) == test_user_id
        
        keys = api_key_manager.list_keys(test_user_id)
        assert len(keys) == 1
        assert keys[0]["description"] == "Key 2"


class TestKeyIsolation:
    """Tests for key isolation between users."""
    
    def test_keys_isolated_by_user(self, api_key_manager):
        """Test that keys are properly isolated between users."""
        user1_key = api_key_manager.generate_key("user1", "User 1 key")
        user2_key = api_key_manager.generate_key("user2", "User 2 key")
        
        # Each key should validate to its own user
        assert api_key_manager.validate_key(user1_key) == "user1"
        assert api_key_manager.validate_key(user2_key) == "user2"
        
        # Each user should only see their own keys
        user1_keys = api_key_manager.list_keys("user1")
        user2_keys = api_key_manager.list_keys("user2")
        
        assert len(user1_keys) == 1
        assert len(user2_keys) == 1
        assert user1_keys[0]["description"] == "User 1 key"
        assert user2_keys[0]["description"] == "User 2 key"

