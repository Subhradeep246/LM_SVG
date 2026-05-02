"""
Main Training Script (Standard Parameterization)
Trains GPT models on SVG data with configurable architecture and hyperparameters.
"""

import os
import sys
import time
import json
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import GPT, GPTConfig
from model.utils import count_parameters, print_model_summary, get_batch_size_and_grad_accum


class SVGDataLoader:
    """Memory-mapped data loader for tokenized SVG data."""

    def __init__(self, data_path, block_size, batch_size, device='cuda'):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Memory-map the binary data
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        self.n_batches = (self.n_tokens - 1) // (block_size * batch_size)

        print(f"  Loaded {data_path}: {self.n_tokens:,} tokens, ~{self.n_batches:,} batches")

    def get_batch(self):
        """Get a random batch of data."""
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(self.device), y.to(self.device)

    def iter_epoch(self):
        """Iterate through the data in order for 1 epoch."""
        total_chunks = (self.n_tokens - 1) // self.block_size
        indices = list(range(0, total_chunks * self.block_size, self.block_size))
        np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue

            x = torch.stack([
                torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64))
                for i in batch_indices
            ])
            y = torch.stack([
                torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64))
                for i in batch_indices
            ])
            yield x.to(self.device), y.to(self.device)


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # Cosine decay
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters=50):
    """Estimate validation loss over multiple batches."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = data_loader.get_batch()
        with autocast(dtype=torch.float16):
            _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_model(
    model_name,
    model_config,
    train_config,
    device='cuda',
    use_wandb=True,
):
    """
    Train a single model configuration.

    Args:
        model_name: Name of the model (e.g., 'tiny', 'small')
        model_config: Dict with n_layer, n_head, n_embd, d_ff
        train_config: Dict with training hyperparameters
        device: Device to train on
        use_wandb: Whether to log to W&B

    Returns:
        results: Dict with training results
    """
    # Extract config values
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
    model_ckpt_dir = os.path.join(checkpoint_dir, f"sp_{model_name}")
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data loaders
    print(f"\nLoading data...")
    batch_size, grad_accum = get_batch_size_and_grad_accum(
        model_config, target_batch_tokens=65536, block_size=block_size
    )
    print(f"  Micro batch size: {batch_size}, Gradient accumulation: {grad_accum}")

    train_loader = SVGDataLoader(
        os.path.join(data_dir, 'train.bin'), block_size, batch_size, device
    )
    val_loader = SVGDataLoader(
        os.path.join(data_dir, 'val.bin'), block_size, batch_size, device
    )

    # Calculate max_iters for 1 epoch
    tokens_per_step = batch_size * grad_accum * block_size
    max_iters = (train_loader.n_tokens // tokens_per_step) * epochs
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Max iterations ({epochs} epoch): {max_iters:,}")

    # Model
    print(f"\nInitializing model: {model_name}")
    model = GPT.from_config_dict(model_config, vocab_size=vocab_size,
                                  block_size=block_size, dropout=dropout)
    model = model.to(device)
    param_info = print_model_summary(model, model_name)

    # Optimizer
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.dim() >= 2],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.dim() < 2],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate,
                                   betas=(beta1, beta2), fused=False)
    scaler = GradScaler()

    # W&B
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_config.get('wandb_project', 'svg-scaling-laws'),
                name=f"sp_{model_name}",
                config={
                    'model_name': model_name,
                    'model_config': model_config,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'grad_accum': grad_accum,
                    'max_iters': max_iters,
                    'params': param_info['non_embedding'],
                    'parameterization': 'standard',
                },
            )
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")
            use_wandb = False

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {model_name} (SP) | LR={learning_rate:.1e} | {max_iters} iters")
    print(f"{'='*60}")

    model.train()
    train_losses = []
    val_losses = []
    timing = []
    tokens_processed = 0
    best_val_loss = float('inf')
    start_time = time.time()

    for it in range(max_iters):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(it, warmup_iters, max_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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

        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = tokens_per_step / dt
        tokens_processed += tokens_per_step

        train_losses.append({'iter': it, 'loss': accum_loss, 'lr': lr})

        # Logging
        if it % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  iter {it:>6d}/{max_iters} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | {elapsed:.0f}s elapsed")

            if use_wandb:
                import wandb
                wandb.log({
                    'train/loss': accum_loss,
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/tokens_processed': tokens_processed,
                    'train/iter': it,
                })

        # Evaluation
        if it % eval_interval == 0 or it == max_iters - 1:
            val_loss = estimate_loss(model, val_loader)
            val_losses.append({'iter': it, 'val_loss': val_loss})
            print(f"  >>> val_loss: {val_loss:.4f}")

            if use_wandb:
                import wandb
                wandb.log({'val/loss': val_loss, 'train/iter': it})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best checkpoint
                ckpt_path = os.path.join(model_ckpt_dir, 'best.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model_config,
                    'iter': it,
                    'val_loss': val_loss,
                    'model_name': model_name,
                }, ckpt_path)

        # Periodic save
        if it > 0 and it % save_interval == 0:
            ckpt_path = os.path.join(model_ckpt_dir, f'iter_{it}.pt')
            torch.save({
                'model': model.state_dict(),
                'config': model_config,
                'iter': it,
            }, ckpt_path)

        timing.append({'iter': it, 'dt': dt, 'tokens_per_sec': tokens_per_sec})

    # Final evaluation
    final_val_loss = estimate_loss(model, val_loader, eval_iters=100)
    total_time = time.time() - start_time

    # Save final checkpoint
    ckpt_path = os.path.join(model_ckpt_dir, 'final.pt')
    torch.save({
        'model': model.state_dict(),
        'config': model_config,
        'iter': max_iters,
        'val_loss': final_val_loss,
        'model_name': model_name,
    }, ckpt_path)

    # GPU memory
    gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0

    # Results
    results = {
        'model_name': model_name,
        'parameterization': 'standard',
        'params_non_embedding': param_info['non_embedding'],
        'params_total': param_info['total'],
        'learning_rate': learning_rate,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'total_time_seconds': total_time,
        'tokens_processed': tokens_processed,
        'avg_tokens_per_sec': tokens_processed / total_time,
        'gpu_memory_gb': gpu_mem,
        'max_iters': max_iters,
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    # Save results
    results_path = os.path.join(results_dir, f"sp_{model_name}_results.json")
    # Save without the full loss curves (too large for JSON)
    results_summary = {k: v for k, v in results.items()
                       if k not in ('train_losses', 'val_losses')}
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save full loss curves as numpy
    np.save(os.path.join(results_dir, f"sp_{model_name}_train_losses.npy"),
            np.array([(l['iter'], l['loss']) for l in train_losses]))
    np.save(os.path.join(results_dir, f"sp_{model_name}_val_losses.npy"),
            np.array([(l['iter'], l['val_loss']) for l in val_losses]))

    print(f"\n{'='*60}")
    print(f"Training Complete: {model_name}")
    print(f"{'='*60}")
    print(f"  Final val loss:      {final_val_loss:.4f}")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Total time:          {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Avg tokens/sec:      {tokens_processed/total_time:.0f}")
    print(f"  GPU memory peak:     {gpu_mem:.2f} GB")

    if use_wandb:
        import wandb
        wandb.finish()

    # Clear GPU memory
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return results


def train_all_models(train_config_path, model_config_path, model_names=None):
    """
    Train all model sizes for the scaling study.

    Args:
        train_config_path: Path to training config YAML
        model_config_path: Path to model configs YAML
        model_names: List of model names to train (None = all)
    """
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    with open(model_config_path, 'r') as f:
        model_configs = yaml.safe_load(f)

    if model_names is None:
        model_names = list(model_configs.keys())

    all_results = {}
    for name in model_names:
        print(f"\n{'#'*60}")
        print(f"# Training model: {name.upper()}")
        print(f"{'#'*60}")

        results = train_model(
            model_name=name,
            model_config=model_configs[name],
            train_config=train_config,
            use_wandb=train_config.get('use_wandb', True),
        )
        all_results[name] = results

    # Save combined results
    results_dir = train_config['results_dir']
    summary = {}
    for name, r in all_results.items():
        summary[name] = {
            'params': r['params_non_embedding'],
            'final_val_loss': r['final_val_loss'],
            'best_val_loss': r['best_val_loss'],
            'total_time_s': r['total_time_seconds'],
            'tokens_per_sec': r['avg_tokens_per_sec'],
            'gpu_memory_gb': r['gpu_memory_gb'],
        }

    summary_path = os.path.join(results_dir, "sp_scaling_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("All Models Trained — Summary")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Params':>10} {'Val Loss':>10} {'Time':>10} {'Tok/s':>10}")
    print(f"{'-'*50}")
    for name, s in summary.items():
        print(f"{name:<10} {s['params']:>10,} {s['final_val_loss']:>10.4f} "
              f"{s['total_time_s']:>8.0f}s {s['tokens_per_sec']:>10.0f}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_configs.yaml")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Specific models to train (e.g., tiny small)")
    args = parser.parse_args()

    train_all_models(args.train_config, args.model_config, args.models)
