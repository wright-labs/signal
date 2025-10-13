"""Training loop utilities."""
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def compute_forward_backward(
    model: Any,
    batch: Dict[str, torch.Tensor],
    accumulate: bool = False,
    loss_fn: str = "causal_lm",
    loss_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute forward and backward pass with custom loss function.
    
    This function:
    1. Moves batch to model device
    2. Computes loss using specified loss function
    3. Performs backward pass to compute gradients
    4. Collects gradient statistics
    """
    from modal_runtime.loss_functions import compute_loss
    
    if loss_kwargs is None:
        loss_kwargs = {}
    
    # Move batch to device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Verify we have trainable parameters
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found! Cannot compute gradients. "
            "This usually means LoRA adapters were not applied correctly."
        )
    
    # Compute loss using specified loss function
    loss, loss_metrics = compute_loss(model, batch, loss_fn=loss_fn, **loss_kwargs)
    
    # Verify loss requires grad
    if not loss.requires_grad:
        raise RuntimeError(
            f"Loss does not require grad! This means gradient flow is broken. "
            f"Loss tensor: {loss}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}. "
            f"This is likely due to missing prepare_model_for_kbit_training() for quantized models."
        )
    
    # Backward pass
    loss.backward()
    
    # Compute gradient statistics
    grad_norms = []
    grad_values = []
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            # Only collect sample values to avoid memory issues
            if len(grad_values) < 1000:
                grad_values.extend(param.grad.flatten().tolist()[:100])
    
    if params_with_grad == 0:
        raise RuntimeError(
            "Backward pass completed but no parameters have gradients! "
            "This should never happen after loss.backward(). Check model setup."
        )
    
    # Calculate gradient statistics
    total_grad_norm = sum(n**2 for n in grad_norms) ** 0.5  # L2 norm
    
    grad_stats = {
        "grad_norm": total_grad_norm,
        "grad_max": max(grad_values) if grad_values else 0.0,
        "grad_min": min(grad_values) if grad_values else 0.0,
        "grad_mean": sum(grad_values) / len(grad_values) if grad_values else 0.0,
        "params_with_grad": params_with_grad,
    }
    
    print(f"Forward-backward complete: loss={loss.item():.4f}, grad_norm={total_grad_norm:.4f}")
    
    # Merge loss metrics and gradient stats
    all_metrics = {**loss_metrics, **grad_stats}
    
    return loss.item(), all_metrics


def save_gradients(
    model: Any,
    save_path: str,
):
    """Save model gradients to disk.
    
    This is useful for the distributed training API where gradient computation
    and optimizer updates happen in separate function calls.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.cpu()
    
    if len(gradients) == 0:
        raise RuntimeError("No gradients to save! Call backward() first.")
    
    torch.save(gradients, save_path)
    print(f"✓ Saved {len(gradients)} gradient tensors to {save_path}")


def load_gradients(
    model: Any,
    load_path: str,
):
    """Load gradients into model."""
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Gradients not found: {load_path}")
    
    gradients = torch.load(load_path, map_location="cpu")
    
    loaded_count = 0
    for name, param in model.named_parameters():
        if name in gradients:
            if param.grad is None:
                param.grad = gradients[name].to(param.device)
            else:
                param.grad.add_(gradients[name].to(param.device))
            loaded_count += 1
    
    print(f"✓ Loaded {loaded_count} gradient tensors from {load_path}")

