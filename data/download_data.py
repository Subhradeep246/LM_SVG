"""
SVG Data Download Script
Downloads SVG datasets from HuggingFace and saves to Google Drive.
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm


def download_all_datasets(output_dir):
    """Download all SVG datasets from HuggingFace."""
    os.makedirs(output_dir, exist_ok=True)

    datasets_info = {
        "svg-icons-simple": {
            "name": "starvector/svg-icons-simple",
            "description": "~89,370 simplified SVG icons (primary dataset)",
            "priority": "primary",
        },
        "svg-emoji-simple": {
            "name": "starvector/svg-emoji-simple",
            "description": "Simplified SVG emoji (~14.5 MB)",
            "priority": "supplementary",
        },
        "svg-fonts-simple": {
            "name": "starvector/svg-fonts-simple",
            "description": "Simplified SVG font glyphs (~2.38 GB) - subsampled",
            "priority": "supplementary",
            "max_samples": 150000,
        },
    }

    all_stats = {}

    for key, info in datasets_info.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"{'='*60}")

        try:
            ds = load_dataset(info["name"], split="train")
            total = len(ds)
            print(f"  Total samples available: {total:,}")

            # Subsample if needed
            max_samples = info.get("max_samples", None)
            if max_samples and total > max_samples:
                print(f"  Subsampling to {max_samples:,} samples...")
                ds = ds.shuffle(seed=42).select(range(max_samples))
                total = max_samples

            # Finding the SVG column
            svg_column = None
            for col in ds.column_names:
                if "svg" in col.lower():
                    svg_column = col
                    break
            if svg_column is None:
                # Trying common column names
                for col in ["text", "content", "code"]:
                    if col in ds.column_names:
                        svg_column = col
                        break
            if svg_column is None:
                svg_column = ds.column_names[0]

            print(f"  Using column: '{svg_column}'")

            # Saving as JSONL
            output_path = os.path.join(output_dir, f"{key}.jsonl")
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for item in tqdm(ds, desc=f"  Saving {key}", total=total):
                    svg_text = item[svg_column]
                    if isinstance(svg_text, str) and len(svg_text.strip()) > 0:
                        json.dump({"svg": svg_text, "source": key}, f, ensure_ascii=False)
                        f.write("\n")
                        count += 1

            all_stats[key] = {
                "total_available": total,
                "saved": count,
                "file": output_path,
                "priority": info["priority"],
            }
            print(f"  Saved {count:,} SVGs to {output_path}")

        except Exception as e:
            print(f"  ERROR downloading {info['name']}: {e}")
            all_stats[key] = {"error": str(e)}

    # Saved download stats
    stats_path = os.path.join(output_dir, "download_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    total_svgs = sum(s.get("saved", 0) for s in all_stats.values())
    print(f"Total SVGs downloaded: {total_svgs:,}")
    for key, stats in all_stats.items():
        if "error" not in stats:
            print(f"  {key}: {stats['saved']:,} SVGs")
        else:
            print(f"  {key}: FAILED - {stats['error']}")

    return all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/raw")
    args = parser.parse_args()
    download_all_datasets(args.output_dir)
