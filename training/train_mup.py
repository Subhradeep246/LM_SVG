"""
µP Training Script
Trains GPT models with Maximal Update Parameterization for LR transfer.
"""

import os
import sys
import time
import json
import math
import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer_mup import MuPGPT, MuPGPTConfig, setup_mup, get_mup_optimizer, MUP_AVAILABLE
from model.utils import print_model_summary, get_batch_size_and_grad_accum
from training.train import SVGDataLoader, get_lr, estimate_loss


def train_mup_model(
    model_name,
    model_config,
    base_config,  # Config of the smallest (base) model for µP
    train_config,
    device='cuda',
    use_wandb=True,
):
    """
    Train a single model with µP parameterization.

    Args:
        model_name: Name of the model
        model_config: Dict with architecture config
        base_config: Dict with base (smallest) model config for µP shapes
        train_config: Dict with training hyperparameters
        device: Training device
        use_wandb: Whether to log to W&B
    """
    if not MUP_AVAILABLE:
        raise ImportError("mup package is required. Install with: pip install mup")

    # Config values
    data_dir = train_config['data_dir']
    checkpoint_dir = train_config['checkpoint_dir']
    results_dir = train_config['results_dir']
    block_size = train_config['block_size']
    vocab_size = train_config['vocab_size']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 0.1)
    beta1 = train_config.get('beta1', 0.9)
    beta2 = train_config.get('beta2', 0.95)
    grad_clip = train_config.get('grad_clip', 1.0)
    warmup_iters = train_config.get('warmup_iters', 200)
    min_lr_ratio = train_config.get('min_lr_ratio', 0.1)
    log_interval = train_config.get('log_interval', 50)
    eval_interval = train_config.get('eval_interval', 200)
    save_interval = train_config.get('save_interval', 500)
    dropout = train_config.get('dropout', 0.0)
    epochs = train_config.get('epochs', 1)

    min_lr = learning_rate * min_lr_ratio

    # Directories
    model_ckpt_dir = os.path.join(checkpoint_dir, f"mup_{model_name}")
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data
    # Force a stricter memory limit (3000MB estimated) to avoid OOM on Large/XL
    batch_size, grad_accum = get_batch_size_and_grad_accum(
        model_config, target_batch_tokens=65536, block_size=block_size, max_memory_mb=3000
    )
    print(f"  Micro batch size: {batch_size}, Gradient accumulation: {grad_accum}")

    train_loader = SVGDataLoader(
        os.path.join(data_dir, 'train.bin'), block_size, batch_size, device
    )
    val_loader = SVGDataLoader(
        os.path.join(data_dir, 'val.bin'), block_size, batch_size, device
    )

    tokens_per_step = batch_size * grad_accum * block_size
    max_iters = (train_loader.n_tokens // tokens_per_step) * epochs

    # Model with µP
    print(f"\nInitializing µP model: {model_name}")
    model = MuPGPT.from_config_dict(model_config, vocab_size=vocab_size,
                                     block_size=block_size, dropout=dropout)

    # Set µP base shapes
    model = setup_mup(model, base_config, vocab_size=vocab_size, block_size=block_size)
    model = model.to(device)
    param_info = print_model_summary(model, f"{model_name} (µP)")

    # µP optimizer
    optimizer = get_mup_optimizer(model, lr=learning_rate,
                                  weight_decay=weight_decay, betas=(beta1, beta2))
    scaler = GradScaler()

    # W&B
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_config.get('wandb_project', 'svg-scaling-laws'),
                name=f"mup_{model_name}",
                config={
                    'model_name': model_name,
                    'model_config': model_config,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'params': param_info['non_embedding'],
                    'parameterization': 'mup',
                },
            )
        except Exception as e:
            print(f"W&B init failed: {e}")
            use_wandb = False

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {model_name} (µP) | LR={learning_rate:.1e} | {max_iters} iters")
    print(f"{'='*60}")

    model.train()
    train_losses = []
    val_losses = []
    tokens_processed = 0
    best_val_loss = float('inf')
    start_time = time.time()

    # µP: Capture per-group base LRs set by MuAdamW (these contain
    # µP's per-layer multipliers). We apply the cosine schedule as a
    # ratio so that the multipliers are preserved.
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    for it in range(max_iters):
        t0 = time.time()

        # LR schedule — apply as ratio to preserve µP per-layer multipliers
        lr = get_lr(it, warmup_iters, max_iters, learning_rate, min_lr)
        lr_ratio = lr / learning_rate  # ratio relative to base LR
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg['lr'] = base_lr * lr_ratio

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            X, Y = train_loader.get_batch()
            with autocast(dtype=torch.float16):
                _, loss = model(X, Y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0
        tokens_per_sec = tokens_per_step / dt
        tokens_processed += tokens_per_step

        train_losses.append({'iter': it, 'loss': accum_loss, 'lr': lr})

        if it % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  iter {it:>6d}/{max_iters} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | {elapsed:.0f}s")

            if use_wandb:
                import wandb
                wandb.log({
                    'train/loss': accum_loss, 'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec, 'train/iter': it,
                })

        if it % eval_interval == 0 or it == max_iters - 1:
            val_loss = estimate_loss(model, val_loader)
            val_losses.append({'iter': it, 'val_loss': val_loss})
            print(f"  >>> val_loss: {val_loss:.4f}")

            if use_wandb:
                import wandb
                wandb.log({'val/loss': val_loss, 'train/iter': it})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'config': model_config,
                    'iter': it, 'val_loss': val_loss,
                    'model_name': model_name,
                    'parameterization': 'mup',
                }, os.path.join(model_ckpt_dir, 'best.pt'))

    # Final
    final_val_loss = estimate_loss(model, val_loader, eval_iters=100)
    total_time = time.time() - start_time
    gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0

    torch.save({
        'model': model.state_dict(),
        'config': model_config,
        'val_loss': final_val_loss,
        'model_name': model_name,
        'parameterization': 'mup',
    }, os.path.join(model_ckpt_dir, 'final.pt'))

    results = {
        'model_name': model_name,
        'parameterization': 'mup',
        'params_non_embedding': param_info['non_embedding'],
        'params_total': param_info['total'],
        'learning_rate': learning_rate,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'total_time_seconds': total_time,
        'tokens_processed': tokens_processed,
        'avg_tokens_per_sec': tokens_processed / total_time,
        'gpu_memory_gb': gpu_mem,
    }

    lr_str = f"{learning_rate:.0e}".replace("+", "").replace("0", "")
    if lr_str.endswith('-'): lr_str = lr_str[:-1] # clean up format like 1e-2

    results_path = os.path.join(results_dir, f"mup_{model_name}_lr{lr_str}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(results_dir, f"mup_{model_name}_lr{lr_str}_train_losses.npy"),
            np.array([(l['iter'], l['loss']) for l in train_losses]))
    np.save(os.path.join(results_dir, f"mup_{model_name}_lr{lr_str}_val_losses.npy"),
            np.array([(l['iter'], l['val_loss']) for l in val_losses]))

    print(f"\nTraining Complete: {model_name} (µP) [LR: {learning_rate}]")
    print(f"  Final val loss: {final_val_loss:.4f} | Best: {best_val_loss:.4f}")
    print(f"  Time: {total_time:.0f}s | GPU: {gpu_mem:.2f} GB")

    if use_wandb:
        import wandb
        wandb.finish()

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return results


def train_all_mup_models(train_config_path, model_config_path, model_names=None):
    """Train all model sizes with µP."""
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    with open(model_config_path, 'r') as f:
        model_configs = yaml.safe_load(f)

    if model_names is None:
        model_names = list(model_configs.keys())

    # Base config is always 'tiny'
    base_config = model_configs['tiny']

    all_results = {}
    for name in model_names:
        print(f"\n{'#'*60}")
        print(f"# Training µP model: {name.upper()}")
        print(f"{'#'*60}")

        results = train_mup_model(
            model_name=name,
            model_config=model_configs[name],
            base_config=base_config,
            train_config=train_config,
            use_wandb=train_config.get('use_wandb', True),
        )
        all_results[name] = results

    # Save summary
    results_dir = train_config['results_dir']
    summary = {name: {
        'params': r['params_non_embedding'],
        'final_val_loss': r['final_val_loss'],
        'best_val_loss': r['best_val_loss'],
        'total_time_s': r['total_time_seconds'],
    } for name, r in all_results.items()}

    # Format LR string for filename
    learning_rate = train_config.get('learning_rate', 1e-2)
    lr_str = f"{learning_rate:.0e}".replace("+", "").replace("0", "")
    if lr_str.endswith('-'): lr_str = lr_str[:-1]

    summary_file = f"mup_scaling_summary_lr{lr_str}.json"
    with open(os.path.join(results_dir, summary_file), 'w') as f:
        json.dump(summary, f, indent=2)

    return all_results
