"""
Comprehensive Evaluation Script
Computes perplexity, validity metrics, and render rates for generated SVGs.
"""

import os
import sys
import json
import math
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.validate_svg import validate_batch
from evaluation.render_svg import render_batch


def compute_perplexity(model, data_path, block_size=1024, batch_size=16,
                       device='cuda', max_batches=None):
    """
    Compute perplexity on a dataset.

    Args:
        model: Trained GPT model
        data_path: Path to binary token file
        block_size: Context window
        batch_size: Evaluation batch size
        device: Computation device
        max_batches: Limit evaluation to N batches (None = all)

    Returns:
        perplexity: e^(avg cross-entropy loss)
    """
    model.eval()
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    n_tokens = len(data)
    n_chunks = (n_tokens - 1) // block_size

    total_loss = 0.0
    total_count = 0

    indices = list(range(0, n_chunks * block_size, block_size))
    if max_batches:
        indices = indices[:max_batches * batch_size]

    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc="Computing perplexity"):
            batch_indices = indices[start:start + batch_size]
            if len(batch_indices) == 0:
                continue

            x = torch.stack([
                torch.from_numpy(data[i:i + block_size].astype(np.int64))
                for i in batch_indices
            ]).to(device)
            y = torch.stack([
                torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
                for i in batch_indices
            ]).to(device)

            with autocast(dtype=torch.float16):
                logits, loss = model(x, y)

            total_loss += loss.item() * len(batch_indices)
            total_count += len(batch_indices)

    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def evaluate_generated_svgs(svg_texts, output_dir):
    """
    Comprehensive evaluation of generated SVG samples.

    Args:
        svg_texts: List of generated SVG strings
        output_dir: Directory to save results and renders

    Returns:
        metrics: Dict with all evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating {len(svg_texts)} generated SVGs")
    print(f"{'='*60}")

    # 1. Validity metrics
    print("\n1. Checking validity...")
    validity = validate_batch(svg_texts)
    print(f"  XML valid:         {validity['xml_valid']}/{validity['total']} "
          f"({validity['xml_valid_rate']:.1%})")
    print(f"  SVG root:          {validity['svg_root']}/{validity['total']} "
          f"({validity['svg_root_rate']:.1%})")
    print(f"  Closed tags:       {validity['closed_tags']}/{validity['total']} "
          f"({validity['closed_tags_rate']:.1%})")
    print(f"  Valid attributes:  {validity['valid_attributes']}/{validity['total']} "
          f"({validity['valid_attributes_rate']:.1%})")
    print(f"  Structurally valid:{validity['structurally_valid']}/{validity['total']} "
          f"({validity['structurally_valid_rate']:.1%})")

    # 2. Render rate
    print("\n2. Rendering SVGs...")
    render_dir = os.path.join(output_dir, 'renders')
    render_results = render_batch(svg_texts, render_dir)
    render_success = sum(1 for _, s, _ in render_results if s)
    render_rate = render_success / len(svg_texts) if len(svg_texts) > 0 else 0
    print(f"  Render success:    {render_success}/{len(svg_texts)} ({render_rate:.1%})")

    # 3. Length statistics
    lengths = [len(svg) for svg in svg_texts]
    length_stats = {
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
    }

    # Compile metrics
    metrics = {
        'total_samples': len(svg_texts),
        'validity': validity,
        'render_rate': render_rate,
        'render_success': render_success,
        'length_stats': length_stats,
    }

    # Saving metrics
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Saving SVG texts
    with open(os.path.join(output_dir, 'generated_svgs.json'), 'w') as f:
        json.dump(svg_texts, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return metrics


def full_evaluation(model, test_data_path, generated_svgs, output_dir,
                    block_size=1024, device='cuda'):
    """
    Run complete evaluation: perplexity + generation quality.

    Args:
        model: Trained model
        test_data_path: Path to test.bin
        generated_svgs: List of generated SVG strings
        output_dir: Output directory
        block_size: Context window
        device: Device

    Returns:
        all_metrics: Combined metrics dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # Computing test set perplexity
    print("Computing test set perplexity...")
    perplexity, avg_loss = compute_perplexity(
        model, test_data_path, block_size=block_size, device=device
    )
    print(f"  Test perplexity: {perplexity:.2f}")
    print(f"  Test loss:       {avg_loss:.4f}")

    # Generation evaluation
    gen_metrics = evaluate_generated_svgs(generated_svgs, output_dir)

    # Combined
    all_metrics = {
        'test_perplexity': perplexity,
        'test_loss': avg_loss,
        **gen_metrics,
    }

    with open(os.path.join(output_dir, 'full_evaluation.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Test Perplexity:       {perplexity:.2f}")
    print(f"  XML Validity Rate:     {gen_metrics['validity']['xml_valid_rate']:.1%}")
    print(f"  SVG Render Rate:       {gen_metrics['render_rate']:.1%}")
    print(f"  Structural Valid Rate: {gen_metrics['validity']['structurally_valid_rate']:.1%}")

    return all_metrics
