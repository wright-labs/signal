"""Supported models configuration."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional


class ModelRegistry:
    """Registry of supported models."""
    
    def __init__(self, config_file: str = "config/models.yaml"):
        """Initialize the model registry."""
        if not Path(config_file).is_absolute():
            module_dir = Path(__file__).parent.parent
            self.config_file = module_dir / config_file
        else:
            self.config_file = Path(config_file)
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, Dict]:
        """Load models from configuration file."""
        if not self.config_file.exists():
            return {}
        
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
        
        models_dict = {}
        for model in config.get("models", []):
            models_dict[model["name"]] = model
        
        return models_dict
    
    def get_model(self, model_name: str) -> Optional[Dict]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def is_supported(self, model_name: str) -> bool:
        """Check if a model is supported."""
        return model_name in self.models
    
    def list_models(self) -> List[str]:
        """List all supported model names."""
        return list(self.models.keys())

