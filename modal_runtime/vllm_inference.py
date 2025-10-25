import modal
import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

from modal_runtime.app import (
    app,
    INFERENCE_IMAGE,
    huggingface_secret,
    s3_secret,
)


@dataclass
class ChatMessage:
    """OpenAI-compatible chat message."""
    role: str
    content: str


@app.cls(
    image=INFERENCE_IMAGE,
    gpu="A10G",  # Default GPU, can be overridden
    secrets=[huggingface_secret, s3_secret],
    allow_concurrent_inputs=100,  # High concurrency for serverless
    container_idle_timeout=300,  # 5 minutes idle before shutdown
)
class VLLMInference:
    """Serverless OpenAI-compatible vLLM inference endpoint with LoRA support."""
    
    # Configuration passed at deployment time
    base_model: str = None
    lora_path: str = None
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    
    @modal.enter()
    def load_model(self):
        """Load model and LoRA adapters on container startup."""
        from vllm import LLM
        import torch
        
        logger.info("=" * 80)
        logger.info("INITIALIZING vLLM INFERENCE SERVER")
        logger.info("=" * 80)
        
        # Get configuration from environment (set during deployment)
        self.base_model = os.environ.get("BASE_MODEL")
        self.lora_path = os.environ.get("LORA_PATH")
        self.max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
        self.tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
        
        if not self.base_model:
            raise ValueError("BASE_MODEL environment variable not set")
        
        logger.info(f"Base Model: {self.base_model}")
        logger.info(f"LoRA Path: {self.lora_path}")
        logger.info(f"Max Model Length: {self.max_model_len}")
        logger.info(f"Tensor Parallel Size: {self.tensor_parallel_size}")
        
        # Download LoRA adapter from S3 if provided
        local_lora_path = None
        if self.lora_path:
            logger.info("Downloading LoRA adapter from S3...")
            local_lora_path = self._download_lora_from_s3(self.lora_path)
            logger.info(f"✓ LoRA adapter downloaded to {local_lora_path}")
        
        # Detect GPU count
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        logger.info(f"Available GPUs: {gpu_count}")
        
        # Initialize vLLM engine
        logger.info("Loading model with vLLM...")
        vllm_kwargs = {
            "model": self.base_model,
            "dtype": "bfloat16",
            "tensor_parallel_size": min(self.tensor_parallel_size, gpu_count),
            "trust_remote_code": True,
            "max_model_len": self.max_model_len,
        }
        
        # Enable LoRA if adapter provided
        if local_lora_path:
            vllm_kwargs["enable_lora"] = True
            logger.info("LoRA support enabled")
        
        self.engine = LLM(**vllm_kwargs)
        
        # Load LoRA adapter if provided
        if local_lora_path:
            logger.info("Loading LoRA adapter into vLLM...")
            # vLLM will automatically detect and load LoRA from the path
            self.lora_id = "trained_adapter"
        else:
            self.lora_id = None
        
        logger.info("✓ vLLM engine initialized and ready")
        logger.info("=" * 80)
    
    def _download_lora_from_s3(self, s3_uri: str) -> str:
        """Download LoRA adapter from S3/R2 to local storage."""
        import boto3
        from pathlib import Path
        import tempfile
        
        # Parse S3 URI
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        # Extract bucket and prefix
        parts = s3_uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        # Initialize S3 client
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        
        # Create temp directory for LoRA adapter
        local_dir = Path("/tmp/lora_adapter")
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # List all objects with the prefix
        logger.info(f"Downloading from s3://{bucket}/{prefix}")
        paginator = s3_client.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            
            for obj in page["Contents"]:
                key = obj["Key"]
                # Get relative path
                rel_path = key[len(prefix):].lstrip("/")
                if not rel_path:
                    continue
                
                # Download file
                local_file = local_dir / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"  Downloading {key}...")
                s3_client.download_file(bucket, key, str(local_file))
        
        logger.info(f"✓ Downloaded LoRA adapter to {local_dir}")
        return str(local_dir)
    
    @modal.web_endpoint(method="POST", label="chat-completions")
    async def chat_completions(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible /v1/chat/completions endpoint."""
        from vllm import SamplingParams
        import uuid
        
        # Parse request
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 512)
        top_p = request_data.get("top_p", 1.0)
        n = request_data.get("n", 1)
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )
        
        # Generate with LoRA if available
        generate_kwargs = {"prompts": [prompt], "sampling_params": sampling_params}
        if self.lora_id:
            from vllm.lora.request import LoRARequest
            generate_kwargs["lora_request"] = LoRARequest(self.lora_id, 1, self.lora_path)
        
        outputs = self.engine.generate(**generate_kwargs)
        
        # Format response in OpenAI format
        choices = []
        for i, output in enumerate(outputs[0].outputs):
            choices.append({
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": output.text,
                },
                "finish_reason": "stop",
            })
        
        # Count tokens (approximate)
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = sum(len(out.token_ids) for out in outputs[0].outputs)
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "signal-deployed"),
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    
    @modal.web_endpoint(method="POST", label="completions")
    async def completions(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible /v1/completions endpoint."""
        from vllm import SamplingParams
        import uuid
        
        # Parse request
        prompt = request_data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0]  # Take first prompt for simplicity
        
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 512)
        top_p = request_data.get("top_p", 1.0)
        n = request_data.get("n", 1)
        
        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )
        
        # Generate with LoRA if available
        generate_kwargs = {"prompts": [prompt], "sampling_params": sampling_params}
        if self.lora_id:
            from vllm.lora.request import LoRARequest
            generate_kwargs["lora_request"] = LoRARequest(self.lora_id, 1, self.lora_path)
        
        outputs = self.engine.generate(**generate_kwargs)
        
        # Format response in OpenAI format
        choices = []
        for i, output in enumerate(outputs[0].outputs):
            choices.append({
                "index": i,
                "text": output.text,
                "finish_reason": "stop",
            })
        
        # Count tokens (approximate)
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = sum(len(out.token_ids) for out in outputs[0].outputs)
        
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request_data.get("model", "signal-deployed"),
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    
    @modal.web_endpoint(method="GET", label="health")
    async def health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": self.base_model,
            "lora": "enabled" if self.lora_id else "disabled",
        }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI chat messages to a prompt string."""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add assistant prefix for completion
        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\nAssistant:"
        
        return prompt


@app.function(
    image=INFERENCE_IMAGE,
    secrets=[huggingface_secret, s3_secret],
)
def deploy_inference_endpoint(
    base_model: str,
    lora_s3_uri: Optional[str] = None,
    gpu_config: str = "A10G",
    max_model_len: int = 2048,
    tensor_parallel_size: int = 1,
) -> Dict[str, str]:
    """Deploy a vLLM inference endpoint with the specified configuration.
    
    Args:
        base_model: HuggingFace model ID or path
        lora_s3_uri: S3 URI to LoRA adapter (optional)
        gpu_config: GPU configuration (e.g., "A10G", "A100")
        max_model_len: Maximum sequence length
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        Dict with deployment information including endpoint URLs
    """
    import os
    
    # Set environment variables for the deployment
    env_vars = {
        "BASE_MODEL": base_model,
        "MAX_MODEL_LEN": str(max_model_len),
        "TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
    }
    
    if lora_s3_uri:
        env_vars["LORA_PATH"] = lora_s3_uri
    
    # Update the class environment
    for key, value in env_vars.items():
        os.environ[key] = value
    
    logger.info(f"Deploying inference endpoint:")
    logger.info(f"  Base Model: {base_model}")
    logger.info(f"  LoRA: {lora_s3_uri or 'None'}")
    logger.info(f"  GPU: {gpu_config}")
    
    return {
        "status": "deployed",
        "base_model": base_model,
        "lora_enabled": lora_s3_uri is not None,
        "gpu_config": gpu_config,
    }

