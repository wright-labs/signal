"""Manifest generation for training artifacts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA256 hash of training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        SHA256 hash as hex string with 'sha256:' prefix
    """
    # Create a stable JSON representation (sorted keys)
    config_json = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.sha256(config_json.encode('utf-8'))
    return f"sha256:{hash_obj.hexdigest()}"


def get_library_versions() -> Dict[str, str]:
    """Get versions of key ML libraries.
    
    Returns:
        Dict mapping library names to version strings
    """
    versions = {}
    
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass
    
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import peft
        versions["peft"] = peft.__version__
    except ImportError:
        pass
    
    try:
        import bitsandbytes
        versions["bitsandbytes"] = bitsandbytes.__version__
    except ImportError:
        pass
    
    try:
        import accelerate
        versions["accelerate"] = accelerate.__version__
    except ImportError:
        pass
    
    return versions


def scan_artifact_files(artifact_path: str) -> tuple[List[str], Dict[str, int]]:
    """Scan artifact directory and collect file information.
    
    Args:
        artifact_path: Path to artifact directory
        
    Returns:
        Tuple of (file_list, file_sizes_dict)
    """
    artifact_dir = Path(artifact_path)
    
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact path not found: {artifact_path}")
    
    files = []
    file_sizes = {}
    
    # Collect all files recursively
    for file_path in artifact_dir.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(artifact_dir)
            relative_str = str(relative_path).replace('\\', '/')
            files.append(relative_str)
            file_sizes[relative_str] = file_path.stat().st_size
    
    # Sort for consistency
    files.sort()
    
    return files, file_sizes


def generate_manifest(
    run_id: str,
    owner_id: str,
    step: int,
    run_config: Dict[str, Any],
    artifact_path: str,
    mode: str = "adapter",
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate manifest for training artifact.
    
    Args:
        run_id: Run identifier
        owner_id: Owner/tenant identifier
        step: Training step number
        run_config: Full run configuration
        artifact_path: Path to artifact directory
        mode: Artifact mode ('adapter', 'merged', 'state')
        metrics: Optional training metrics (loss, grad_norm, etc.)
        
    Returns:
        Manifest dictionary
    """
    # Scan artifact files
    files, file_sizes = scan_artifact_files(artifact_path)
    
    # Compute config hash
    config_hash = compute_config_hash(run_config)
    
    # Get library versions
    library_versions = get_library_versions()
    
    # Extract LoRA config
    lora_config = {
        "r": run_config.get("lora_r"),
        "alpha": run_config.get("lora_alpha"),
        "dropout": run_config.get("lora_dropout"),
        "target_modules": run_config.get("lora_target_modules"),
    }
    
    # Build manifest
    manifest = {
        "version": "1.0",
        "run_id": run_id,
        "owner_id": owner_id,
        "step": step,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": run_config.get("base_model"),
        "framework": run_config.get("framework"),
        "lora_config": lora_config,
        "config_hash": config_hash,
        "files": files,
        "file_sizes": file_sizes,
        "total_size_bytes": sum(file_sizes.values()),
        "library_versions": library_versions,
    }
    
    # Add training metrics if provided
    if metrics:
        manifest["training_metrics"] = metrics
    
    # Add GPU config
    if "gpu_config" in run_config:
        manifest["gpu_config"] = {
            "type": run_config.get("gpu_type"),
            "count": run_config.get("gpu_count"),
            "config": run_config.get("gpu_config"),
        }
    
    # Add training hyperparameters
    manifest["training_config"] = {
        "optimizer": run_config.get("optimizer"),
        "learning_rate": run_config.get("learning_rate"),
        "weight_decay": run_config.get("weight_decay"),
        "max_seq_length": run_config.get("max_seq_length"),
        "bf16": run_config.get("bf16"),
        "gradient_checkpointing": run_config.get("gradient_checkpointing"),
        "load_in_8bit": run_config.get("load_in_8bit"),
        "load_in_4bit": run_config.get("load_in_4bit"),
    }
    
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> bool:
    """Validate that manifest has all required fields.
    
    Args:
        manifest: Manifest dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        "version",
        "run_id",
        "owner_id",
        "step",
        "timestamp",
        "base_model",
        "config_hash",
        "files",
        "file_sizes",
    ]
    
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Missing required field in manifest: {field}")
    
    # Validate files and file_sizes match
    if set(manifest["files"]) != set(manifest["file_sizes"].keys()):
        raise ValueError("Files list and file_sizes keys don't match")
    
    return True


def save_manifest_to_file(manifest: Dict[str, Any], output_path: str) -> None:
    """Save manifest to JSON file.
    
    Args:
        manifest: Manifest dictionary
        output_path: Path where manifest.json should be saved
    """
    validate_manifest(manifest)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_manifest_from_file(manifest_path: str) -> Dict[str, Any]:
    """Load manifest from JSON file.
    
    Args:
        manifest_path: Path to manifest.json
        
    Returns:
        Manifest dictionary
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    validate_manifest(manifest)
    return manifest


def verify_config_compatibility(
    checkpoint_manifest: Dict[str, Any],
    current_config: Dict[str, Any]
) -> bool:
    """Verify that checkpoint config is compatible with current config.
    
    Used during resume to ensure we're loading a compatible checkpoint.
    
    Args:
        checkpoint_manifest: Manifest from saved checkpoint
        current_config: Current run configuration
        
    Returns:
        True if compatible, raises ValueError if incompatible
    """
    checkpoint_hash = checkpoint_manifest.get("config_hash")
    current_hash = compute_config_hash(current_config)
    
    if checkpoint_hash != current_hash:
        # Configs don't match - check if differences are acceptable
        critical_fields = [
            "base_model",
            "framework",
            "lora_r",
            "lora_alpha",
            "max_seq_length",
        ]
        
        checkpoint_config = checkpoint_manifest.get("training_config", {})
        
        for field in critical_fields:
            checkpoint_value = checkpoint_config.get(field)
            current_value = current_config.get(field)
            
            if checkpoint_value != current_value:
                raise ValueError(
                    f"Incompatible checkpoint: {field} mismatch "
                    f"(checkpoint: {checkpoint_value}, current: {current_value})"
                )
    
    return True

