"""
Learning Rate Sweep Script (Standard Parameterization)
Performs LR sweep on the smallest model to find optimal learning rate.
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train_model


def lr_sweep(
    model_config,
    train_config,
    model_name='tiny',
    lr_values=None,
    use_wandb=True,
):
    """
    Perform a learning rate sweep on a single model.

    Args:
        model_config: Dict with model architecture
        train_config: Dict with training config (will override LR)
        model_name: Name of the model
        lr_values: List of learning rates to try
        use_wandb: Whether to log to W&B

    Returns:
        sweep_results: Dict mapping LR -> val_loss
    """
    if lr_values is None:
        lr_values = train_config.get('lr_sweep_values',
                                      [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])

    results_dir = train_config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Learning Rate Sweep: {model_name}")
    print(f"Testing {len(lr_values)} learning rates: {lr_values}")
    print(f"{'='*60}")

    sweep_results = {}

    for lr in lr_values:
        print(f"\n--- LR = {lr:.1e} ---")

        # Override LR in config
        config = train_config.copy()
        config['learning_rate'] = lr

        try:
            result = train_model(
                model_name=f"{model_name}_lr{lr:.0e}",
                model_config=model_config,
                train_config=config,
                use_wandb=use_wandb,
            )

            sweep_results[str(lr)] = {
                'learning_rate': lr,
                'final_val_loss': result['final_val_loss'],
                'best_val_loss': result['best_val_loss'],
                'total_time_seconds': result['total_time_seconds'],
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            sweep_results[str(lr)] = {
                'learning_rate': lr,
                'error': str(e),
            }

    # Find best LR
    valid_results = {k: v for k, v in sweep_results.items() if 'error' not in v}
    best_lr_key = min(valid_results, key=lambda k: valid_results[k]['final_val_loss'])
    best_lr = valid_results[best_lr_key]['learning_rate']
    best_loss = valid_results[best_lr_key]['final_val_loss']

    print(f"\n{'='*60}")
    print(f"LR Sweep Results:")
    print(f"{'='*60}")
    print(f"{'LR':>12} {'Val Loss':>12} {'Status':>10}")
    print(f"{'-'*40}")
    for k, v in sweep_results.items():
        if 'error' in v:
            print(f"{v['learning_rate']:>12.1e} {'N/A':>12} {'FAILED':>10}")
        else:
            marker = " <<<" if v['learning_rate'] == best_lr else ""
            print(f"{v['learning_rate']:>12.1e} {v['final_val_loss']:>12.4f} {'OK':>10}{marker}")
    print(f"\nBest LR: {best_lr:.1e} (val_loss = {best_loss:.4f})")

    # Save results
    sweep_path = os.path.join(results_dir, f"sp_lr_sweep_{model_name}.json")
    with open(sweep_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'parameterization': 'standard',
            'best_lr': best_lr,
            'best_val_loss': best_loss,
            'results': sweep_results,
        }, f, indent=2)

    # Plot
    plot_lr_sweep(sweep_results, results_dir, model_name, 'sp')

    return best_lr, sweep_results


def plot_lr_sweep(sweep_results, output_dir, model_name, prefix='sp'):
    """Plot learning rate sweep results."""
    lrs = []
    losses = []
    for k, v in sweep_results.items():
        if 'error' not in v:
            lrs.append(v['learning_rate'])
            losses.append(v['final_val_loss'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(lrs, losses, 'o-', color='#E74C3C', markersize=10, linewidth=2, markeredgecolor='#2C3E50')

    # Highlight best
    best_idx = np.argmin(losses)
    ax.plot(lrs[best_idx], losses[best_idx], 's', color='#2ECC71', markersize=15,
            markeredgecolor='#2C3E50', markeredgewidth=2, zorder=5,
            label=f'Best: LR={lrs[best_idx]:.1e}, Loss={losses[best_idx]:.4f}')

    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    param_label = 'Standard' if prefix == 'sp' else 'µP'
    ax.set_title(f'Learning Rate Sweep — {model_name} ({param_label})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_lr_sweep_{model_name}.png'), dpi=150)
    plt.close()
    print(f"Saved LR sweep plot to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_configs.yaml")
    parser.add_argument("--model_name", type=str, default="tiny")
    args = parser.parse_args()

    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_configs = yaml.safe_load(f)

    best_lr, results = lr_sweep(
        model_config=model_configs[args.model_name],
        train_config=train_config,
        model_name=args.model_name,
    )
    print(f"\nBest learning rate: {best_lr}")
