"""
Dataset Preparation Script
Tokenizes cleaned SVGs, creates train/val/test splits, and saves as binary files.
"""

import os
import json
import numpy as np
from tqdm import tqdm


def prepare_dataset(
    cleaned_path,
    tokenizer_dir,
    output_dir,
    block_size=1024,
    train_ratio=0.98,
    val_ratio=0.01,
    test_ratio=0.01,
    max_token_length=4096,
    seed=42,
):
    """
    Preparing tokenized dataset for training.

    1. Load cleaned SVGs
    2. Tokenize each with BPE
    3. Filter by token length
    4. Split into train/val/test by file
    5. Concatenate with <eos> separators
    6. Chunk into fixed-length sequences
    7. Save as memory-mapped numpy arrays

    Args:
        cleaned_path: Path to cleaned JSONL
        tokenizer_dir: Directory with trained tokenizer
        output_dir: Directory to save binary files
        block_size: Context window size
        train_ratio/val_ratio/test_ratio: Split ratios
        max_token_length: Maximum tokens per SVG (filter threshold)
        seed: Random seed for reproducible splitting
    """
    from data.train_tokenizer import load_tokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Loading tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_dir)
    eos_id = tokenizer.token_to_id("<eos>")
    bos_id = tokenizer.token_to_id("<bos>")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  EOS token ID: {eos_id}")
    print(f"  BOS token ID: {bos_id}")

    # Loading and tokenizing SVGs
    print(f"\nLoading and tokenizing SVGs from {cleaned_path}...")
    all_tokenized = []
    lengths = []
    filtered_count = 0

    with open(cleaned_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        try:
            item = json.loads(line.strip())
            svg = item.get("svg", "")
        except json.JSONDecodeError:
            continue

        # Tokenize
        encoded = tokenizer.encode(svg)
        token_ids = encoded.ids

        # Filter by token length
        if len(token_ids) > max_token_length:
            filtered_count += 1
            continue

        all_tokenized.append(token_ids)
        lengths.append(len(token_ids))

    print(f"\nTokenized {len(all_tokenized):,} SVGs")
    print(f"Filtered (>{max_token_length} tokens): {filtered_count:,}")
    print(f"Token length stats:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Std: {np.std(lengths):.1f}")

    # Split by file index (not by token position)
    np.random.seed(seed)
    n = len(all_tokenized)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    split_stats = {}

    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_indices):,} files)...")

        # Concatenating with EOS separators
        all_tokens = []
        for idx in tqdm(split_indices, desc=f"  Concatenating {split_name}"):
            all_tokens.extend(all_tokenized[idx])
            all_tokens.append(eos_id)

        total_tokens = len(all_tokens)
        print(f"  Total tokens: {total_tokens:,}")

        # Saving as numpy binary
        arr = np.array(all_tokens, dtype=np.uint16)  # uint16 supports vocab up to 65535
        bin_path = os.path.join(output_dir, f"{split_name}.bin")
        arr.tofile(bin_path)
        print(f"  Saved to: {bin_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)")

        split_stats[split_name] = {
            "num_files": len(split_indices),
            "total_tokens": total_tokens,
            "num_chunks": total_tokens // block_size,
            "file_size_mb": os.path.getsize(bin_path) / 1e6,
        }

    # Saving length distribution for plotting
    lengths_path = os.path.join(output_dir, "token_lengths.npy")
    np.save(lengths_path, np.array(lengths))

    # Saving preparation stats
    stats = {
        "total_files": len(all_tokenized),
        "filtered_files": filtered_count,
        "max_token_length": max_token_length,
        "block_size": block_size,
        "vocab_size": tokenizer.get_vocab_size(),
        "splits": split_stats,
        "length_stats": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "std": float(np.std(lengths)),
        },
    }

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("Dataset Preparation Summary:")
    print(f"{'='*60}")
    print(f"Block size: {block_size}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    for split_name, ss in split_stats.items():
        print(f"\n{split_name}:")
        print(f"  Files: {ss['num_files']:,}")
        print(f"  Tokens: {ss['total_tokens']:,}")
        print(f"  Chunks (block_size={block_size}): {ss['num_chunks']:,}")

    # Checking minimum token requirement
    train_tokens = split_stats["train"]["total_tokens"]
    if train_tokens >= 100_000_000:
        print(f"\n Training tokens ({train_tokens:,}) >= 100M requirement")
    else:
        print(f"\n Training tokens ({train_tokens:,}) < 100M. Consider adding more data!")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_path", type=str,
                        default="/content/drive/MyDrive/svg-scaling/data/cleaned.jsonl")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/tokenizer")
    parser.add_argument("--output_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/data")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--max_token_length", type=int, default=2048)
    args = parser.parse_args()

    prepare_dataset(
        args.cleaned_path, args.tokenizer_dir, args.output_dir,
        block_size=args.block_size, max_token_length=args.max_token_length,
    )
