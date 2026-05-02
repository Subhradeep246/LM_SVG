"""
Standard Parameterization vs µP Comparison
Comparison plots and analysis.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.scaling_plots import fit_power_law, power_law


def compare_sp_vs_mup(results_dir, output_dir, mup_lr="1e-2"):
    """
    Compare standard parameterization and µP scaling curves.

    Args:
        results_dir: Directory with all training results
        output_dir: Directory to save comparison plots
        mup_lr: The specific µP learning rate to load and compare
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load SP results
    sp_path = os.path.join(results_dir, "sp_scaling_summary.json")
    with open(sp_path, 'r') as f:
        sp_summary = json.load(f)

    # Load specific µP results based on LR
    mup_path = os.path.join(results_dir, f"mup_scaling_summary_lr{mup_lr}.json")
    if not os.path.exists(mup_path):
        print(f"Warning: Could not find {mup_path}, trying generic name.")
        mup_path = os.path.join(results_dir, "mup_scaling_summary.json")
        
    with open(mup_path, 'r') as f:
        mup_summary = json.load(f)

    model_names = list(sp_summary.keys())

    sp_params = [sp_summary[n]['params'] for n in model_names]
    sp_losses = [sp_summary[n]['final_val_loss'] for n in model_names]
    mup_params = [mup_summary[n]['params'] for n in model_names]
    mup_losses = [mup_summary[n]['final_val_loss'] for n in model_names]

    # Fit power laws
    sp_popt, sp_pcov, sp_r2 = fit_power_law(sp_params, sp_losses)
    mup_popt, mup_pcov, mup_r2 = fit_power_law(mup_params, mup_losses)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    # SP data + fit
    ax.scatter(sp_params, sp_losses, s=120, c='#E74C3C', edgecolors='#2C3E50',
               linewidth=2, zorder=5, marker='o', label='Standard Param.')
    if sp_popt is not None:
        a, alpha, c = sp_popt
        x_fit = np.geomspace(min(sp_params) * 0.5, max(sp_params) * 2, 100)
        y_fit = power_law(x_fit, *sp_popt)
        ax.plot(x_fit, y_fit, '--', color='#E74C3C', linewidth=2, alpha=0.7,
                label=f'SP fit: α={alpha:.3f}, R²={sp_r2:.3f}')

    # µP data + fit
    ax.scatter(mup_params, mup_losses, s=120, c='#3498DB', edgecolors='#2C3E50',
               linewidth=2, zorder=5, marker='s', label='µP')
    if mup_popt is not None:
        a, alpha, c = mup_popt
        x_fit = np.geomspace(min(mup_params) * 0.5, max(mup_params) * 2, 100)
        y_fit = power_law(x_fit, *mup_popt)
        ax.plot(x_fit, y_fit, '--', color='#3498DB', linewidth=2, alpha=0.7,
                label=f'µP fit: α={alpha:.3f}, R²={mup_r2:.3f}')

    # Labels for each model
    for i, name in enumerate(model_names):
        ax.annotate(name, (sp_params[i], sp_losses[i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color='#E74C3C', alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Non-Embedding Parameters', fontsize=14)
    ax.set_ylabel('Validation Loss (1 epoch)', fontsize=14)
    ax.set_title('Scaling Laws: Standard Parameterization vs µP', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sp_vs_mup_scaling_lr{mup_lr}.png'), dpi=150)
    plt.close()
    print(f"Saved comparison plot to sp_vs_mup_scaling_lr{mup_lr}.png")

    # Improvement analysis
    print(f"\n{'='*60}")
    print("SP vs µP Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'SP Loss':>10} {'µP Loss':>10} {'Δ':>10} {'Improve':>10}")
    print(f"{'-'*50}")
    for name in model_names:
        sp_loss = sp_summary[name]['final_val_loss']
        mup_loss = mup_summary[name]['final_val_loss']
        delta = sp_loss - mup_loss
        improve = delta / sp_loss * 100
        print(f"{name:<10} {sp_loss:>10.4f} {mup_loss:>10.4f} {delta:>+10.4f} {improve:>+9.1f}%")

    if sp_popt is not None and mup_popt is not None:
        print(f"\nScaling exponent (α):")
        print(f"  SP:  {sp_popt[1]:.4f}")
        print(f"  µP:  {mup_popt[1]:.4f}")
        print(f"  Improvement: {((mup_popt[1]/sp_popt[1]) - 1)*100:+.1f}%")

    # Saving comparison results
    comparison = {
        'sp': {
            'alpha': float(sp_popt[1]) if sp_popt is not None else None,
            'r_squared': float(sp_r2) if sp_r2 is not None else None,
            'losses': {n: sp_summary[n]['final_val_loss'] for n in model_names},
        },
        'mup': {
            'alpha': float(mup_popt[1]) if mup_popt is not None else None,
            'r_squared': float(mup_r2) if mup_r2 is not None else None,
            'losses': {n: mup_summary[n]['final_val_loss'] for n in model_names},
        },
    }
    with open(os.path.join(output_dir, f'sp_vs_mup_comparison_lr{mup_lr}.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def plot_lr_sweep_comparison(results_dir, output_dir):
    """Plot SP and µP learning rate sweeps side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, (prefix, title, color) in enumerate([
        ('sp', 'Standard Parameterization', '#E74C3C'),
        ('mup', 'µP Parameterization', '#3498DB'),
    ]):
        sweep_path = os.path.join(results_dir, f"{prefix}_lr_sweep_tiny.json")
        if not os.path.exists(sweep_path):
            continue

        with open(sweep_path, 'r') as f:
            sweep = json.load(f)

        lrs = []
        losses = []
        for k, v in sweep['results'].items():
            if 'error' not in v:
                lrs.append(v['learning_rate'])
                losses.append(v['final_val_loss'])

        axes[i].semilogx(lrs, losses, 'o-', color=color, markersize=10, linewidth=2,
                         markeredgecolor='#2C3E50')

        best_idx = np.argmin(losses)
        axes[i].plot(lrs[best_idx], losses[best_idx], 's', color='#2ECC71',
                     markersize=15, markeredgecolor='#2C3E50', markeredgewidth=2,
                     zorder=5, label=f'Best: {lrs[best_idx]:.1e}')

        axes[i].set_xlabel('Learning Rate', fontsize=12)
        axes[i].set_ylabel('Validation Loss', fontsize=12)
        axes[i].set_title(f'LR Sweep — {title}', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=11)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_sweep_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved LR sweep comparison plot")
