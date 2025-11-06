"""Modal deployment entry point - imports all classes to register them."""

# Import app first
from modal_runtime.app import app

# Import all classes decorated with @app.cls() to register them
# These imports register the classes with the app
from modal_runtime.training_session import TrainingSession  # noqa: F401
from modal_runtime.vllm_inference import VLLMInference  # noqa: F401

# The app is now ready for deployment with all classes registered
if __name__ == "__main__":
    app.deploy()

