"""Modal app definition with images and volumes.

This module defines:
- Modal app configuration
- Docker images for training and inference
- Persistent volumes for data storage
- Secrets for API keys and credentials
"""
from modal import App, Image as ModalImage, Volume, Secret
from pathlib import Path

# Configuration

HOURS = 60 * 60 # this is the default timeout for the modal app

# Modal app, volumes, and secrets

app = App("signal")

# Shared volume for all runs
data_volume = Volume.from_name("signal-data", create_if_missing=True)

VOLUME_CONFIG = {
    "/data": data_volume,
}

# Secrets
huggingface_secret = Secret.from_name("secrets-hf-wandb")
s3_secret = Secret.from_name("aws-s3-credentials")
api_secret = Secret.from_name("signal-api-secrets")

# Docker image configurations

# CUDA configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Training image with PyTorch, Transformers, PEFT for LoRA training
# Streamlined to include only essential dependencies for transformers + PEFT
TRAINING_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
    )
    # Install PyTorch and core dependencies
    .uv_pip_install(
        [
            "torch",
            "torchvision", 
            "torchaudio",
        ]
    )
    # Install transformers ecosystem and LoRA training dependencies
    .uv_pip_install(
        [
            "transformers",           # HuggingFace transformers
            "peft",                   # Parameter-Efficient Fine-Tuning (LoRA)
            "bitsandbytes",           # Quantization and 8-bit optimizer
            "accelerate",             # Training utilities
            "safetensors",            # Safe model serialization
            "sentencepiece",          # Tokenization
            "protobuf",               # Protocol buffers
            "datasets",               # HuggingFace datasets
            "nvidia-ml-py3",          # GPU monitoring (pynvml)
            "wandb",                  # Experiment tracking
            "boto3",                  # AWS S3 for artifact storage
            "botocore",               # AWS core library
            "hf_transfer",            # Fast HuggingFace downloads
        ]
    )
    # Set environment variables for optimization
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster HF downloads
            "HF_HOME": "/data/.cache",         # Cache models on volume
            "PYTHONPATH": "/root",             # For local module imports
        }
    )
    # Add modal_runtime as a local directory for imports
    .add_local_dir(
        local_path=Path(__file__).parent,
        remote_path="/root/modal_runtime",
    )
)

# TODO: wait check this one
# Inference image - use same as training for simplicity
# Previously tried vLLM but it caused OOM during build
# The training image works fine for inference too
INFERENCE_IMAGE = TRAINING_IMAGE

# TODO: also validate why this is imported at the end
# Import training session classes to register stateful container classes
# This must be at the end after all images and secrets are defined
# import modal_runtime.training_session  # noqa: F401, E402  # Single class (disabled in favor of multi_gpu_session)
import modal_runtime.multi_gpu_session  # noqa: F401, E402  # Multiple GPU config classes
