"""Modal runtime package - imports all classes for deployment."""

# Import app
from modal_runtime.app import app

# Import classes to register them with the app
# These imports are needed for Modal to discover and register the classes
# The actual imports will work in Modal's environment even if they fail locally

try:
    from modal_runtime.training_session import TrainingSession  # noqa: F401
except ImportError:
    # This is expected if dependencies aren't installed locally
    # Modal will handle the imports during deployment
    pass

try:
    from modal_runtime.vllm_inference import VLLMInference  # noqa: F401
except ImportError:
    pass

__all__ = ['app']

