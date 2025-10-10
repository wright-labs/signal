"""Modal app definition with images and volumes.

NOTE: All ML dependencies (torch, transformers, peft, etc.) are installed
in the Modal Docker images below. They are NOT in requirements-api.txt since
the API server only needs FastAPI and client libraries.
"""
from modal import App, Image as ModalImage, Volume, Secret
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

HOURS = 60 * 60

# =============================================================================
# MODAL APP, VOLUMES, AND SECRETS
# =============================================================================

app = App("signal")

# Shared volume for all runs
data_volume = Volume.from_name("signal-data", create_if_missing=True)

VOLUME_CONFIG = {
    "/data": data_volume,
}

# HuggingFace and WandB secrets
huggingface_secret = Secret.from_name("secrets-hf-wandb")

# AWS S3 credentials for artifact storage
s3_secret = Secret.from_name("aws-s3-credentials")

# API secrets for internal service-to-service communication
# This should contain: SIGNAL_INTERNAL_SECRET, SIGNAL_API_URL
api_secret = Secret.from_name("signal-api-secrets")

# =============================================================================
# DOCKER IMAGE CONFIGURATIONS
# =============================================================================

CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Training image with PyTorch, Transformers, PEFT, Axolotl and multi-GPU support
# Add modal_runtime as a local directory so it can be imported as a package
TRAINING_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
    )
    # Install PyTorch and core dependencies using uv for better performance
    .uv_pip_install(
        [
            "torch",
            "torchvision",
            "torchaudio",
        ]
    )
    # Install build dependencies
    .run_commands(
        "uv pip install --no-deps -U packaging setuptools wheel ninja --system"
    )
    # Install Axolotl with DeepSpeed for multi-GPU support
    .run_commands("uv pip install --no-build-isolation axolotl[deepspeed] --system")
    # Install flash-attention for efficient attention computation
    .run_commands(
        "UV_NO_BUILD_ISOLATION=1 uv pip install flash-attn --no-build-isolation --system"
    )
    # Additional dependencies for compatibility
    .uv_pip_install(
        [
            "sentencepiece",
            "protobuf",
            "safetensors",
            "pyyaml",
            "wandb",
            "requests",
            "boto3",  # S3 storage
            "botocore",
        ]
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable faster HF downloads
            "HF_HOME": "/data/.cache",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir(
        local_path=Path(__file__).parent,
        remote_path="/root/modal_runtime",
    )
)

# Inference image - simplified to use same image as training
# Using vLLM and flash-attn causes OOM during image build, so we use the training image
# which already has transformers and can do inference just fine
INFERENCE_IMAGE = TRAINING_IMAGE

