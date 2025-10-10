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

# =============================================================================
# DOCKER IMAGE CONFIGURATIONS
# =============================================================================

CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Training image with PyTorch, Transformers, PEFT, and Unsloth
# Add modal_runtime as a local directory so it can be imported as a package
TRAINING_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
    )
    .run_commands("pip install --upgrade pip")
    .pip_install(
        [
            "torch",
            "torchvision", 
            "torchaudio",
            "transformers",
            "accelerate",
            "peft",
            "bitsandbytes",
            "datasets",
            "xformers",
            "trl",
            "sentencepiece",
            "protobuf",
            "safetensors",
            "pyyaml",
            "wandb",
            "requests",
        ]
    )
    .env(
        {
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

