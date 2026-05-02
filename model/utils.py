"""
Model Utilities
Parameter counting, memory estimation, config loading.
"""

import yaml
import torch


def count_parameters(model, non_embedding=True):
    """Count model parameters (optionally excluding embeddings)."""
    total = sum(p.numel() for p in model.parameters())
    if non_embedding and hasattr(model, 'transformer'):
        emb_params = model.transformer.wte.weight.numel()
        emb_params += model.transformer.wpe.weight.numel()
        return total - emb_params
    return total


def estimate_memory_mb(config, batch_size=1, block_size=1024, dtype_bytes=2):
    """
    Estimate GPU memory usage in MB.

    Args:
        config: dict with n_layer, n_head, n_embd, d_ff
        batch_size: Batch size
        block_size: Sequence length
        dtype_bytes: Bytes per parameter (2 for FP16, 4 for FP32)
    """
    n_embd = config['n_embd']
    n_layer = config['n_layer']
    d_ff = config['d_ff']
    vocab_size = 4096

    # Parameter memory
    # Embeddings
    params = vocab_size * n_embd + block_size * n_embd
    # Attention: Q, K, V projections + output projection per layer
    params += n_layer * (4 * n_embd * n_embd)
    # FFN: up + down projections per layer
    params += n_layer * (n_embd * d_ff + d_ff * n_embd)
    # LayerNorms: 2 per layer
    params += n_layer * (4 * n_embd)
    # Output head
    params += n_embd * vocab_size

    param_memory = params * dtype_bytes / 1e6

    # Activation memory (rough estimate)
    # Per layer: attention scores + FFN activations
    act_per_layer = batch_size * block_size * n_embd * 4  # approximate
    activation_memory = n_layer * act_per_layer * dtype_bytes / 1e6

    # Optimizer states (Adam: 2x param memory for momentum + variance)
    optimizer_memory = params * 4 * 2 / 1e6  # Always FP32 for optimizer states

    total = param_memory + activation_memory + optimizer_memory

    return {
        'params': params,
        'param_memory_mb': param_memory,
        'activation_memory_mb': activation_memory,
        'optimizer_memory_mb': optimizer_memory,
        'total_memory_mb': total,
    }


def estimate_flops_per_step(config, batch_size, block_size):
    """
    Estimate FLOPs per training step (forward + backward ≈ 3× forward).

    Based on: FLOPs ≈ 6 * N * B * T (for a forward+backward pass)
    where N = non-embedding params, B = batch size, T = sequence length
    """
    n_embd = config['n_embd']
    n_layer = config['n_layer']
    d_ff = config['d_ff']

    # Non-embedding params (approximate)
    N = n_layer * (4 * n_embd**2 + 2 * n_embd * d_ff + 4 * n_embd) + n_embd * 4096

    # FLOPs ≈ 6 * N * B * T
    flops = 6 * N * batch_size * block_size

    return flops


def load_model_configs(config_path):
    """Load model configurations from YAML file."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs


def get_batch_size_and_grad_accum(config, target_batch_tokens=65536, block_size=1024, max_memory_mb=10000):
    """
    Determine batch size and gradient accumulation steps based on memory constraints.

    Args:
        config: Model config dict
        target_batch_tokens: Target tokens per optimization step
        block_size: Sequence length
        max_memory_mb: Available GPU memory in MB (T4 ≈ 15GB, leave some headroom)

    Returns:
        batch_size: Micro-batch size
        grad_accum_steps: Gradient accumulation steps
    """
    target_batch_size = target_batch_tokens // block_size

    # Binary search for max batch size that fits in memory
    for bs in [64, 32, 16, 8, 4, 2, 1]:
        mem = estimate_memory_mb(config, batch_size=bs, block_size=block_size)
        if mem['total_memory_mb'] < max_memory_mb:
            batch_size = bs
            break
    else:
        batch_size = 1

    grad_accum_steps = max(1, target_batch_size // batch_size)

    return batch_size, grad_accum_steps


def print_model_summary(model, config_name="model"):
    """Print a summary of model architecture and parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_emb = model.count_parameters(non_embedding=True) if hasattr(model, 'count_parameters') else total

    print(f"\n{'='*50}")
    print(f"Model: {config_name}")
    print(f"{'='*50}")
    print(f"Total parameters:         {total:>12,}")
    print(f"Trainable parameters:     {trainable:>12,}")
    print(f"Non-embedding parameters: {non_emb:>12,}")
    print(f"{'='*50}")

    return {'total': total, 'trainable': trainable, 'non_embedding': non_emb}
