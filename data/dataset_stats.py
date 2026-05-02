"""
Dataset Statistics and Visualization
Computes and visualizes statistics about the SVG dataset.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def compute_and_plot_stats(data_dir, output_dir):
    """
    dataset statistics and visualizations.

    Args:
        data_dir: Directory with prepared dataset (binary files + stats)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load stats
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # Load token lengths
    lengths_path = os.path.join(data_dir, "token_lengths.npy")
    lengths = np.load(lengths_path)

    print(f"{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total files: {stats['total_files']:,}")
    print(f"Filtered files: {stats['filtered_files']:,}")
    print(f"Vocab size: {stats['vocab_size']}")
    print(f"Block size: {stats['block_size']}")
    print(f"\nLength stats:")
    for k, v in stats['length_stats'].items():
        print(f"  {k}: {v}")
    print(f"\nSplit stats:")
    for split, ss in stats['splits'].items():
        print(f"  {split}: {ss['num_files']:,} files, {ss['total_tokens']:,} tokens")

    # Plot 1: Token length histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=100, color='#4ECDC4', edgecolor='#2C3E50', alpha=0.8)
    ax.set_xlabel('Token Length', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of SVG Token Lengths', fontsize=16, fontweight='bold')
    ax.axvline(np.mean(lengths), color='#E74C3C', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(lengths):.0f}')
    ax.axvline(np.median(lengths), color='#3498DB', linestyle='--', linewidth=2,
               label=f'Median: {np.median(lengths):.0f}')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_length_distribution.png'), dpi=150)
    plt.close()
    print(f"\nSaved token_length_distribution.png")

    # Plot 2: Token count per split (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    split_names = list(stats['splits'].keys())
    split_tokens = [stats['splits'][s]['total_tokens'] for s in split_names]
    colors = ['#2ECC71', '#F39C12', '#E74C3C']
    bars = ax.bar(split_names, split_tokens, color=colors, edgecolor='#2C3E50', linewidth=1.5)
    for bar, count in zip(bars, split_tokens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(split_tokens)*0.02,
                f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Tokens', fontsize=14)
    ax.set_title('Token Counts per Split', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_token_counts.png'), dpi=150)
    plt.close()
    print(f"Saved split_token_counts.png")

    # Plot 3: CDF of token lengths
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax.plot(sorted_lengths, cdf, color='#9B59B6', linewidth=2)
    ax.set_xlabel('Token Length', fontsize=14)
    ax.set_ylabel('Cumulative Proportion', fontsize=14)
    ax.set_title('CDF of SVG Token Lengths', fontsize=16, fontweight='bold')
    ax.axhline(0.95, color='#E74C3C', linestyle='--', alpha=0.7,
               label='95th percentile')
    p95 = np.percentile(lengths, 95)
    ax.axvline(p95, color='#E74C3C', linestyle='--', alpha=0.7)
    ax.text(p95 + 5, 0.5, f'p95={p95:.0f}', fontsize=12, color='#E74C3C')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_length_cdf.png'), dpi=150)
    plt.close()
    print(f"Saved token_length_cdf.png")

    return stats


def render_example_svgs(cleaned_path, output_dir, num_examples=12):
    """
    Render example SVGs at different complexity levels.

    Args:
        cleaned_path: Path to cleaned JSONL
        output_dir: Directory to save rendered examples
    """
    try:
        import cairosvg
    except ImportError:
        print("CairoSVG not installed. Skipping rendering.")
        return

    from data.train_tokenizer import load_tokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load SVGs and sort by length (proxy for complexity)
    print("Loading SVGs for rendering examples...")
    svgs = []
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                svgs.append(item['svg'])
            except (json.JSONDecodeError, KeyError):
                continue

    # Sort by length
    svgs.sort(key=len)

    # Select examples at different complexity levels
    n = len(svgs)
    indices = np.linspace(0, n - 1, num_examples, dtype=int)

    rendered = []
    for i, idx in enumerate(indices):
        svg = svgs[idx]
        png_path = os.path.join(output_dir, f"example_{i:02d}_len{len(svg)}.png")
        try:
            cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                           write_to=png_path,
                           output_width=200, output_height=200)
            rendered.append(png_path)
            print(f"  Rendered example {i}: {len(svg)} chars → {png_path}")
        except Exception as e:
            print(f"  Failed to render example {i}: {e}")

    # grid of all rendered examples
    if rendered:
        try:
            from PIL import Image
            grid_size = int(np.ceil(np.sqrt(len(rendered))))
            cell_size = 200
            grid = Image.new('RGB', (grid_size * cell_size, grid_size * cell_size), 'white')
            for i, path in enumerate(rendered):
                img = Image.open(path).convert("RGBA").resize((cell_size, cell_size))
                row = i // grid_size
                col = i % grid_size
                grid.paste(img, (col * cell_size, row * cell_size), img)
            grid_path = os.path.join(output_dir, 'example_grid.png')
            grid.save(grid_path)
            print(f"\nSaved example grid: {grid_path}")
        except Exception as e:
            print(f"Failed to create grid: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/data")
    parser.add_argument("--cleaned_path", type=str,
                        default="/content/drive/MyDrive/svg-scaling/data/cleaned.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/results/data_stats")
    args = parser.parse_args()

    compute_and_plot_stats(args.data_dir, args.output_dir)
    render_example_svgs(args.cleaned_path, args.output_dir)
