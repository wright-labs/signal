"""Tests for model registry."""

import tempfile
from pathlib import Path
from api.models import ModelRegistry


class TestModelLoading:
    """Tests for loading models from configuration."""

    def test_load_models_from_config(self, model_registry):
        """Test loading models from YAML config."""
        models = model_registry.list_models()

        assert len(models) > 0
        assert "meta-llama/Llama-3.2-1B" in models
        assert "meta-llama/Llama-3.2-3B" in models
        assert "google/gemma-2-2b" in models

    def test_parse_model_configurations(self, model_registry):
        """Test that model configurations are parsed correctly."""
        model = model_registry.get_model("meta-llama/Llama-3.2-1B")

        assert model is not None
        assert model["name"] == "meta-llama/Llama-3.2-1B"
        assert model["framework"] == "transformers"
        assert model["gpu"] == "l40s:1"
        assert model["family"] == "llama"

    def test_load_missing_config_file(self):
        """Test handling missing config file."""
        registry = ModelRegistry(config_file="nonexistent.yaml")

        # Should not crash, just return empty
        models = registry.list_models()
        assert models == []

    def test_load_multiple_model_families(self, model_registry):
        """Test loading models from different families."""
        llama_model = model_registry.get_model("meta-llama/Llama-3.2-3B")
        gemma_model = model_registry.get_model("google/gemma-2-2b")

        assert llama_model["family"] == "llama"
        assert gemma_model["family"] == "gemma"

        assert llama_model["framework"] == "transformers"
        assert gemma_model["framework"] == "unsloth"


class TestModelQueries:
    """Tests for querying model information."""

    def test_get_model_by_name(self, model_registry):
        """Test getting a model by name."""
        model = model_registry.get_model("meta-llama/Llama-3.2-3B")

        assert model is not None
        assert model["name"] == "meta-llama/Llama-3.2-3B"

    def test_get_nonexistent_model(self, model_registry):
        """Test getting a model that doesn't exist."""
        model = model_registry.get_model("nonexistent/model")
        assert model is None

    def test_is_supported_true(self, model_registry):
        """Test checking if a model is supported."""
        assert model_registry.is_supported("meta-llama/Llama-3.2-1B") is True
        assert model_registry.is_supported("meta-llama/Llama-3.2-3B") is True

    def test_is_supported_false(self, model_registry):
        """Test checking if a model is not supported."""
        assert model_registry.is_supported("nonexistent/model") is False
        assert model_registry.is_supported("") is False

    def test_list_all_models(self, model_registry):
        """Test listing all available models."""
        models = model_registry.list_models()

        assert isinstance(models, list)
        assert len(models) == 3  # Based on test config
        assert "meta-llama/Llama-3.2-1B" in models
        assert "meta-llama/Llama-3.2-3B" in models
        assert "google/gemma-2-2b" in models

    def test_get_gpu_config(self, model_registry):
        """Test getting GPU config for a model."""
        model = model_registry.get_model("meta-llama/Llama-3.2-1B")
        assert model["gpu"] == "l40s:1"

        # Nonexistent model
        assert model_registry.get_model("nonexistent/model") is None

    def test_get_framework(self, model_registry):
        """Test getting framework for a model."""
        llama_model = model_registry.get_model("meta-llama/Llama-3.2-1B")
        gemma_model = model_registry.get_model("google/gemma-2-2b")

        assert llama_model["framework"] == "transformers"
        assert gemma_model["framework"] == "unsloth"

        # Nonexistent model
        assert model_registry.get_model("nonexistent/model") is None

    def test_get_family(self, model_registry):
        """Test getting model family."""
        llama_model = model_registry.get_model("meta-llama/Llama-3.2-1B")
        gemma_model = model_registry.get_model("google/gemma-2-2b")

        assert llama_model["family"] == "llama"
        assert gemma_model["family"] == "gemma"

        # Nonexistent model
        assert model_registry.get_model("nonexistent/model") is None


class TestModelRegistryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_models_list(self):
        """Test registry with no models."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump({"models": []}, f)
            f.flush()
            temp_path = f.name

        try:
            registry = ModelRegistry(config_file=temp_path)
            models = registry.list_models()
            assert models == []
            assert registry.is_supported("any/model") is False
        finally:
            Path(temp_path).unlink()

    def test_models_with_missing_fields(self):
        """Test handling models with missing optional fields."""
        config = {
            "models": [
                {
                    "name": "test/model",
                    "framework": "transformers",
                    "gpu": "l40s:1",
                    # Missing 'family' field
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(config, f)
            f.flush()
            temp_path = f.name

        try:
            registry = ModelRegistry(config_file=temp_path)
            model = registry.get_model("test/model")

            assert model is not None
            assert model["name"] == "test/model"
            assert model.get("family") is None
        finally:
            Path(temp_path).unlink()
