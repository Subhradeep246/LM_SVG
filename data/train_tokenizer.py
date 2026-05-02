"""
BPE Tokenizer Training for SVG Code
Trains a byte-pair encoding tokenizer on the cleaned SVG corpus.
"""

import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def train_bpe_tokenizer(
    data_path,
    output_dir,
    vocab_size=4096,
    min_frequency=2,
):
    """
    Train a BPE tokenizer on SVG data.

    Args:
        data_path: Path to cleaned JSONL file
        output_dir: Directory to save tokenizer
        vocab_size: BPE vocabulary size
        min_frequency: Minimum frequency for BPE merges
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training BPE tokenizer with vocab_size={vocab_size}")
    print(f"Data: {data_path}")

    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: split on whitespace and punctuation, but keep SVG structure
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Extracting SVG texts for training
    print("Loading SVG texts...")
    temp_file = os.path.join(output_dir, "_temp_training_text.txt")

    count = 0
    with open(data_path, 'r', encoding='utf-8') as fin, \
         open(temp_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                item = json.loads(line.strip())
                svg = item.get("svg", "")
                if svg:
                    fout.write(svg + "\n")
                    count += 1
            except json.JSONDecodeError:
                continue

    print(f"Training on {count:,} SVG documents...")

    # Train
    tokenizer.train(files=[temp_file], trainer=trainer)

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # Clean up temp file
    os.remove(temp_file)

    # Print stats
    print(f"\n{'='*60}")
    print(f"Tokenizer trained successfully!")
    print(f"{'='*60}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {tokenizer_path}")

    #  example encodings
    print(f"\nExample encodings:")
    examples = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">',
        '<circle cx="12" cy="12" r="10" fill="#333"/>',
        '<path d="M 10 20 L 30 40 Z" stroke="black"/>',
        '<rect x="0" y="0" width="100" height="100"/>',
        'fill="none" stroke-width="2"',
    ]

    for text in examples:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"\n  Input:   {text}")
        print(f"  Tokens:  {encoded.tokens[:20]}{'...' if len(encoded.tokens) > 20 else ''}")
        print(f"  IDs:     {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
        print(f"  Length:  {len(encoded.ids)} tokens")
        print(f"  Decoded: {decoded[:80]}{'...' if len(decoded) > 80 else ''}")

    # Saving tokenizer info
    info = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
        "training_docs": count,
    }
    info_path = os.path.join(output_dir, "tokenizer_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    return tokenizer


def load_tokenizer(tokenizer_dir):
    """Load a trained tokenizer."""
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    return Tokenizer.from_file(tokenizer_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/content/drive/MyDrive/svg-scaling/data/cleaned.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/tokenizer")
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()

    train_bpe_tokenizer(args.data_path, args.output_dir, args.vocab_size)
