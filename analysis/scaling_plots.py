"""
Scaling Law Plots and Analysis
Fits power laws and creates scaling plots for the report.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def power_law(N, a, alpha, c):
    """Power law function: L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def fit_power_law(params, losses, p0=None):
    """
    Fit a power law L = a * N^(-alpha) + c to the data.

    Args:
        params: Array of parameter counts
        losses: Array of validation losses

    Returns:
        popt: Fitted parameters (a, alpha, c)
        pcov: Covariance matrix
        r_squared: R^2 value
    """
    params = np.array(params, dtype=float)
    losses = np.array(losses, dtype=float)

    if p0 is None:
        p0 = [1.0, 0.5, min(losses) * 0.9]

    try:
        popt, pcov = curve_fit(
            power_law, params, losses, p0=p0,
            maxfev=10000,
            bounds=([0, 0, 0], [np.inf, 5, np.inf])
        )

        # R^2
        predicted = power_law(params, *popt)
        ss_res = np.sum((losses - predicted) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        return popt, pcov, r_squared
    except Exception as e:
        print(f"Power law fit failed: {e}")
        return None, None, None


def plot_scaling_curve(params, losses, model_names, popt=None, output_path=None,
                       title="Transformer Scaling Law", label="Standard"):
    """
    scaling law plot with optional power law fit.

    Args:
        params: Array of parameter counts
        losses: Array of validation losses
        model_names: List of model name labels
        popt: Fitted power law parameters (a, alpha, c)
        output_path: Path to save plot
        title: Plot title
        label: Curve label
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data points
    ax.scatter(params, losses, s=100, c='#E74C3C', edgecolors='#2C3E50',
               linewidth=2, zorder=5, label=f'{label} (data)')

    # Labels
    for i, name in enumerate(model_names):
        ax.annotate(name, (params[i], losses[i]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, color='#2C3E50')

    # Power law fit
    if popt is not None:
        a, alpha, c = popt
        x_fit = np.geomspace(min(params) * 0.5, max(params) * 2, 100)
        y_fit = power_law(x_fit, *popt)
        ax.plot(x_fit, y_fit, '--', color='#E74C3C', linewidth=2, alpha=0.7,
                label=f'{label} fit: L = {a:.2f}·N^(-{alpha:.3f}) + {c:.3f}')

    ax.set_xscale('log')
    ax.set_xlabel('Non-Embedding Parameters', fontsize=14)
    ax.set_ylabel('Validation Loss (1 epoch)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved scaling plot: {output_path}")
    plt.close()


def plot_training_curves(results_dir, model_names, output_path):
    """
    training loss curves for all models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6']

    for i, name in enumerate(model_names):
        color = colors[i % len(colors)]

        # Train loss
        train_path = os.path.join(results_dir, f"sp_{name}_train_losses.npy")
        if os.path.exists(train_path):
            data = np.load(train_path)
            # Subsample for visibility
            step = max(1, len(data) // 500)
            data = data[::step]
            axes[0].plot(data[:, 0], data[:, 1], color=color, alpha=0.8,
                        linewidth=1.5, label=name)

        # Val loss
        val_path = os.path.join(results_dir, f"sp_{name}_val_losses.npy")
        if os.path.exists(val_path):
            data = np.load(val_path)
            axes[1].plot(data[:, 0], data[:, 1], 'o-', color=color,
                        markersize=4, linewidth=1.5, label=name)

    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {output_path}")


def create_model_table(results_dir):
    """
    summary table of all model architectures and results.
    """
    summary_path = os.path.join(results_dir, "sp_scaling_summary.json")
    if not os.path.exists(summary_path):
        print("No scaling summary found.")
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print(f"\n{'='*80}")
    print(f"{'Model':<8} {'Params':>10} {'Val Loss':>10} {'Time (s)':>10} "
          f"{'Tok/s':>10} {'GPU (GB)':>10}")
    print(f"{'-'*68}")
    for name, s in summary.items():
        print(f"{name:<8} {s['params']:>10,} {s['final_val_loss']:>10.4f} "
              f"{s['total_time_s']:>10.0f} {s['tokens_per_sec']:>10.0f} "
              f"{s['gpu_memory_gb']:>10.2f}")
    print(f"{'='*80}")

    return summary


def run_scaling_analysis(results_dir, output_dir):
    """
    scaling analysis: fit power laws, create plots.

    Args:
        results_dir: Directory with training results
        output_dir: Directory to save analysis outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    summary_path = os.path.join(results_dir, "sp_scaling_summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    model_names = list(summary.keys())
    params = [summary[n]['params'] for n in model_names]
    losses = [summary[n]['final_val_loss'] for n in model_names]

    # Fit power law
    print("\nFitting power law: L = a * N^(-α) + c")
    popt, pcov, r2 = fit_power_law(params, losses)

    if popt is not None:
        a, alpha, c = popt
        print(f"  a = {a:.4f}")
        print(f"  α = {alpha:.4f}")
        print(f"  c = {c:.4f}")
        print(f"  R² = {r2:.4f}")

        # Save fit results
        fit_results = {
            'a': float(a), 'alpha': float(alpha), 'c': float(c),
            'r_squared': float(r2),
            'params': [int(p) for p in params],
            'losses': [float(l) for l in losses],
            'model_names': model_names,
        }
        with open(os.path.join(output_dir, 'sp_power_law_fit.json'), 'w') as f:
            json.dump(fit_results, f, indent=2)

    # Create scaling plot
    plot_scaling_curve(
        params, losses, model_names, popt,
        output_path=os.path.join(output_dir, 'sp_scaling_curve.png'),
        title="SVG Transformer Scaling Law (Standard Parameterization)",
    )

    # Training curves
    plot_training_curves(
        results_dir, model_names,
        output_path=os.path.join(output_dir, 'sp_training_curves.png'),
    )

    # Model table
    create_model_table(results_dir)

    return popt, r2
