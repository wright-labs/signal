"""GPU monitoring utilities for training sessions."""
import torch
from typing import Dict, List, Optional
import time


def get_gpu_stats() -> List[Dict]:
    """Get current GPU utilization and memory stats.
    
    Returns:
        List of dicts with stats for each GPU
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_stats = []
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        # Get memory info
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        
        stats = {
            "gpu_id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_allocated_gb": round(mem_allocated, 2),
            "memory_reserved_gb": round(mem_reserved, 2),
            "memory_total_gb": round(mem_total, 2),
            "memory_percent": round((mem_allocated / mem_total) * 100, 1),
        }
        
        # Try to get utilization (requires pynvml)
        try:
            import pynvml
            if not hasattr(get_gpu_stats, '_nvml_initialized'):
                pynvml.nvmlInit()
                get_gpu_stats._nvml_initialized = True
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert mW to W
            
            stats.update({
                "gpu_utilization_percent": util.gpu,
                "memory_utilization_percent": util.memory,
                "temperature_celsius": temp,
                "power_watts": round(power, 1),
            })
        except Exception:
            # pynvml not available or failed
            pass
        
        gpu_stats.append(stats)
    
    return gpu_stats


def get_gpu_summary() -> Dict:
    """Get aggregate GPU stats across all devices.
    
    Returns:
        Dict with summary stats
    """
    stats = get_gpu_stats()
    if not stats:
        return {}
    
    return {
        "num_gpus": len(stats),
        "total_memory_gb": sum(s["memory_total_gb"] for s in stats),
        "total_allocated_gb": sum(s["memory_allocated_gb"] for s in stats),
        "avg_memory_percent": sum(s["memory_percent"] for s in stats) / len(stats),
        "avg_gpu_utilization": sum(s.get("gpu_utilization_percent", 0) for s in stats) / len(stats) if any("gpu_utilization_percent" in s for s in stats) else None,
        "per_gpu": stats,
    }


def print_gpu_stats(stats: Optional[List[Dict]] = None):
    """Print GPU stats in a formatted way.
    
    Args:
        stats: GPU stats from get_gpu_stats(), or None to fetch current
    """
    if stats is None:
        stats = get_gpu_stats()
    
    if not stats:
        print("No GPU stats available")
        return
    
    print(f"\nGPU Stats ({len(stats)} GPU{'s' if len(stats) > 1 else ''}):")
    print("-" * 80)
    
    for s in stats:
        print(f"  GPU {s['gpu_id']}: {s['name']}")
        print(f"    Memory: {s['memory_allocated_gb']:.2f} / {s['memory_total_gb']:.2f} GB ({s['memory_percent']:.1f}%)")
        
        if "gpu_utilization_percent" in s:
            print(f"    Utilization: {s['gpu_utilization_percent']}% GPU, {s['memory_utilization_percent']}% Memory")
        
        if "temperature_celsius" in s:
            print(f"    Temperature: {s['temperature_celsius']}°C")
        
        if "power_watts" in s:
            print(f"    Power: {s['power_watts']}W")


def setup_multi_gpu_model(model, strategy: str = "data_parallel"):
    """Wrap model for multi-GPU training.
    
    Args:
        model: PyTorch model
        strategy: "data_parallel" or "distributed"
        
    Returns:
        Wrapped model
    """
    if not torch.cuda.is_available():
        return model
    
    device_count = torch.cuda.device_count()
    
    if device_count <= 1:
        print(f"Only {device_count} GPU available, skipping multi-GPU setup")
        return model
    
    print(f"Setting up model for {device_count} GPUs with {strategy} strategy...")
    
    if strategy == "data_parallel":
        # Simple DataParallel (easiest, but not most efficient)
        model = torch.nn.DataParallel(model)
        print(f"✓ Model wrapped with DataParallel across {device_count} GPUs")
        print(f"  Primary device: cuda:0")
        print(f"  Replica devices: cuda:{list(range(1, device_count))}")
        
    elif strategy == "distributed":
        # DistributedDataParallel (more efficient, but requires more setup)
        # Note: This requires proper distributed initialization
        print("⚠️  DistributedDataParallel requires additional setup")
        print("   For now, falling back to DataParallel")
        model = torch.nn.DataParallel(model)
    
    return model


class GPUMonitor:
    """Context manager for monitoring GPU usage during operations."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.start_stats = None
        self.end_stats = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_stats = get_gpu_stats()
        
        if self.verbose and self.start_stats:
            print(f"\n[{self.name}] GPU stats before:")
            for s in self.start_stats:
                print(f"  GPU {s['gpu_id']}: {s['memory_allocated_gb']:.2f} GB "
                      f"({s['memory_percent']:.1f}%)")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stats = get_gpu_stats()
        elapsed = time.time() - self.start_time
        
        if self.verbose and self.end_stats:
            print(f"\n[{self.name}] GPU stats after ({elapsed:.2f}s):")
            for i, s in enumerate(self.end_stats):
                start_mem = self.start_stats[i]['memory_allocated_gb'] if i < len(self.start_stats) else 0
                mem_delta = s['memory_allocated_gb'] - start_mem
                delta_str = f"+{mem_delta:.2f}" if mem_delta > 0 else f"{mem_delta:.2f}"
                
                print(f"  GPU {s['gpu_id']}: {s['memory_allocated_gb']:.2f} GB "
                      f"({s['memory_percent']:.1f}%) [{delta_str} GB]")
    
    def get_memory_delta(self) -> Dict[int, float]:
        """Get memory change per GPU in GB."""
        if not self.start_stats or not self.end_stats:
            return {}
        
        deltas = {}
        for i in range(min(len(self.start_stats), len(self.end_stats))):
            start_mem = self.start_stats[i]['memory_allocated_gb']
            end_mem = self.end_stats[i]['memory_allocated_gb']
            deltas[i] = end_mem - start_mem
        
        return deltas

