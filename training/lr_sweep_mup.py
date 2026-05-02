"""
µP Learning Rate Sweep Script
Performs LR sweep on the smallest µP model to find optimal learning rate.
"""

import os
import sys
import json
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_mup import train_mup_model
from training.lr_sweep import plot_lr_sweep


def lr_sweep_mup(
    model_config,
    base_config,
    train_config,
    model_name='tiny',
    lr_values=None,
    use_wandb=True,
):
    """Perform a learning rate sweep on the smallest µP model."""
    if lr_values is None:
        lr_values = train_config.get('lr_sweep_values',
                                      [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])

    results_dir = train_config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"µP Learning Rate Sweep: {model_name}")
    print(f"Testing {len(lr_values)} learning rates")
    print(f"{'='*60}")

    sweep_results = {}

    for lr in lr_values:
        print(f"\n--- µP LR = {lr:.1e} ---")
        config = train_config.copy()
        config['learning_rate'] = lr

        try:
            result = train_mup_model(
                model_name=f"{model_name}_lr{lr:.0e}",
                model_config=model_config,
                base_config=base_config,
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
            sweep_results[str(lr)] = {'learning_rate': lr, 'error': str(e)}

    # Find best
    valid = {k: v for k, v in sweep_results.items() if 'error' not in v}
    best_key = min(valid, key=lambda k: valid[k]['final_val_loss'])
    best_lr = valid[best_key]['learning_rate']
    best_loss = valid[best_key]['final_val_loss']

    print(f"\n{'='*60}")
    print(f"µP LR Sweep Results:")
    print(f"{'='*60}")
    for k, v in sweep_results.items():
        if 'error' in v:
            print(f"  LR={v['learning_rate']:.1e}: FAILED")
        else:
            marker = " <<<" if v['learning_rate'] == best_lr else ""
            print(f"  LR={v['learning_rate']:.1e}: val_loss={v['final_val_loss']:.4f}{marker}")
    print(f"\nBest µP LR: {best_lr:.1e} (val_loss={best_loss:.4f})")

    # Save
    with open(os.path.join(results_dir, f"mup_lr_sweep_{model_name}.json"), 'w') as f:
        json.dump({
            'model_name': model_name,
            'parameterization': 'mup',
            'best_lr': best_lr,
            'best_val_loss': best_loss,
            'results': sweep_results,
        }, f, indent=2)

    plot_lr_sweep(sweep_results, results_dir, model_name, 'mup')

    return best_lr, sweep_results


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

    best_lr, results = lr_sweep_mup(
        model_config=model_configs[args.model_name],
        base_config=model_configs['tiny'],
        train_config=train_config,
        model_name=args.model_name,
    )
