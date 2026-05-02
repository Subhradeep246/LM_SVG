"""
Scaling Law Extrapolation
Predicts validation loss for larger models using fitted power laws.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.scaling_plots import fit_power_law, power_law


def extrapolate_scaling_law(results_dir, output_dir, extrapolation_factor=10, mup_lr="1e-2"):
    """
    Extrapolate scaling law to predict loss for larger models.

    Args:
        results_dir: Directory with training results
        output_dir: Directory to save extrapolation results
        extrapolation_factor: Factor by which to scale up largest model
        mup_lr: The µP learning rate to extrapolate (defaults to 1e-2)

    Returns:
        extrapolation: Dict with prediction and confidence intervals
    """
    os.makedirs(output_dir, exist_ok=True)

    # specific µP first, then generic µP, then fall back to SP
    summary = None
    label = ""
    
    paths_to_try = [
        (os.path.join(results_dir, f"mup_scaling_summary_lr{mup_lr}.json"), f'µP (LR={mup_lr})'),
        (os.path.join(results_dir, "mup_scaling_summary.json"), 'µP'),
        (os.path.join(results_dir, "sp_scaling_summary.json"), 'Standard')
    ]
    
    for path, l in paths_to_try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                summary = json.load(f)
            label = l
            break
            
    if summary is None:
        print("No scaling summary found!")
        return None

    model_names = list(summary.keys())
    params = np.array([summary[n]['params'] for n in model_names], dtype=float)
    losses = np.array([summary[n]['final_val_loss'] for n in model_names], dtype=float)

    # Fit power law
    popt, pcov, r2 = fit_power_law(params, losses)
    if popt is None:
        print("Power law fit failed!")
        return None

    a, alpha, c = popt
    print(f"\nUsing {label} scaling law: L = {a:.4f} · N^(-{alpha:.4f}) + {c:.4f}")
    print(f"R² = {r2:.4f}")

    # Extrapolation target
    max_params = int(max(params))
    target_params = max_params * extrapolation_factor
    predicted_loss = power_law(target_params, *popt)

    # Confidence interval from covariance matrix
    # Use error propagation: σ_L² ≈ J^T · Σ · J
    # where J is the Jacobian of L w.r.t. (a, alpha, c)
    if pcov is not None:
        # Jacobian at target point
        J = np.array([
            target_params ** (-alpha),# ∂L/∂a
            -a * target_params ** (-alpha) * np.log(target_params), # ∂L/∂α
            1.0, # ∂L/∂c
        ])
        sigma_L_sq = J @ pcov @ J
        sigma_L = np.sqrt(max(0, sigma_L_sq))
        ci_low = predicted_loss - 1.96 * sigma_L
        ci_high = predicted_loss + 1.96 * sigma_L
    else:
        sigma_L = None
        ci_low = ci_high = predicted_loss

    # Print results
    print(f"\nExtrapolation Results:")
    print(f"  Largest trained model: {max_params:,} params → loss = {losses[-1]:.4f}")
    print(f"  Extrapolation target:  {target_params:,} params ({extrapolation_factor}×)")
    print(f"  Predicted loss:        {predicted_loss:.4f}")
    if sigma_L:
        print(f"  95% CI:                [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Uncertainty (σ):       {sigma_L:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Original data
    ax.scatter(params, losses, s=120, c='#2ECC71', edgecolors='#2C3E50',
               linewidth=2, zorder=5, label=f'{label} (trained)')

    # Labels
    for i, name in enumerate(model_names):
        ax.annotate(name, (params[i], losses[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)

    # Power law fit line
    x_fit = np.geomspace(min(params) * 0.5, target_params * 1.5, 200)
    y_fit = power_law(x_fit, *popt)
    ax.plot(x_fit, y_fit, '--', color='#2ECC71', linewidth=2, alpha=0.7,
            label=f'Fit: α={alpha:.3f}')

    # Extrapolation point
    ax.scatter([target_params], [predicted_loss], s=200, c='#E74C3C',
               edgecolors='#2C3E50', linewidth=2, zorder=5, marker='*',
               label=f'Extrapolated: {predicted_loss:.4f}')

    # Confidence interval
    if sigma_L:
        ax.fill_between(x_fit, power_law(x_fit, *popt) - 1.96 * sigma_L,
                         power_law(x_fit, *popt) + 1.96 * sigma_L,
                         alpha=0.15, color='#2ECC71', label='95% CI')

    # Extrapolation boundary
    ax.axvline(max(params), color='#F39C12', linestyle=':', linewidth=2,
               alpha=0.5, label='Training boundary')

    ax.set_xscale('log')
    ax.set_xlabel('Non-Embedding Parameters', fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    ax.set_title(f'Scaling Law Extrapolation ({extrapolation_factor}× beyond training)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_extrapolation.png'), dpi=150)
    plt.close()
    print(f"Saved extrapolation plot")

    # Save results
    extrapolation = {
        'parameterization': label,
        'fit': {'a': float(a), 'alpha': float(alpha), 'c': float(c), 'r_squared': float(r2)},
        'trained_models': {
            name: {'params': int(params[i]), 'loss': float(losses[i])}
            for i, name in enumerate(model_names)
        },
        'extrapolation': {
            'target_params': int(target_params),
            'factor': extrapolation_factor,
            'predicted_loss': float(predicted_loss),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
            'sigma': float(sigma_L) if sigma_L else None,
        },
    }

    with open(os.path.join(output_dir, f'extrapolation_results_lr{mup_lr}.json'), 'w') as f:
        json.dump(extrapolation, f, indent=2)

    return extrapolation
