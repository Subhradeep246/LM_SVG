"""
SVG Preprocessing Pipeline
Cleans, normalizes, and validates SVG data for training.
"""

import os
import re
import json
from lxml import etree
from tqdm import tqdm


def strip_comments(svg_text):
    """Remove XML/HTML comments from SVG."""
    return re.sub(r'<!--.*?-->', '', svg_text, flags=re.DOTALL)


def strip_processing_instructions(svg_text):
    """Remove XML processing instructions like <?xml ...?>."""
    return re.sub(r'<\?.*?\?>', '', svg_text, flags=re.DOTALL)


def strip_doctype(svg_text):
    """Remove DOCTYPE declarations."""
    return re.sub(r'<!DOCTYPE.*?>', '', svg_text, flags=re.DOTALL | re.IGNORECASE)


def normalize_whitespace(svg_text):
    """Collapse multiple whitespace into single spaces, trim lines."""
    # Collapse multiple spaces/tabs into single space
    svg_text = re.sub(r'[ \t]+', ' ', svg_text)
    # Remove blank lines
    svg_text = re.sub(r'\n\s*\n', '\n', svg_text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in svg_text.strip().split('\n')]
    return '\n'.join(line for line in lines if line)


def normalize_coordinates(svg_text, precision=1):
    """Round floating point numbers in SVG to specified precision."""
    def round_match(match):
        try:
            val = float(match.group(0))
            if val == int(val):
                return str(int(val))
            return f"{val:.{precision}f}"
        except ValueError:
            return match.group(0)

    # Match floating point numbers (including negative)
    # Avoid matching numbers inside URLs or color hex codes
    svg_text = re.sub(r'(?<![#&\w])(-?\d+\.\d+)', round_match, svg_text)
    return svg_text


def remove_metadata_elements(svg_text):
    """Remove metadata, title, desc, and other non-visual elements."""
    for tag in ['metadata', 'title', 'desc']:
        svg_text = re.sub(
            rf'<{tag}[^>]*>.*?</{tag}>',
            '', svg_text, flags=re.DOTALL | re.IGNORECASE
        )
        # Self-closing variants
        svg_text = re.sub(
            rf'<{tag}[^>]*/>', '', svg_text, flags=re.IGNORECASE
        )
    return svg_text


def remove_unnecessary_attributes(svg_text):
    """Remove attributes that don't affect rendering."""
    attrs_to_remove = [
        r'\s+id="[^"]*"',
        r'\s+class="[^"]*"',
        r'\s+data-[a-z-]+="[^"]*"',
        r'\s+xmlns:xlink="[^"]*"',
        r'\s+xml:space="[^"]*"',
        r'\s+version="[^"]*"',
    ]
    for pattern in attrs_to_remove:
        svg_text = re.sub(pattern, '', svg_text)
    return svg_text


def validate_xml(svg_text):
    """Check if SVG parses as valid XML."""
    try:
        etree.fromstring(svg_text.encode('utf-8'))
        return True
    except etree.XMLSyntaxError:
        return False


def has_svg_root(svg_text):
    """Check if the SVG has a proper <svg> root element."""
    try:
        tree = etree.fromstring(svg_text.encode('utf-8'))
        tag = tree.tag
        # Handle namespace
        if '}' in tag:
            tag = tag.split('}')[1]
        return tag.lower() == 'svg'
    except Exception:
        return False


def clean_single_svg(svg_text, precision=1, remove_metadata=True):
    """Apply full cleaning pipeline to a single SVG."""
    # Strip non-essential content
    svg_text = strip_processing_instructions(svg_text)
    svg_text = strip_doctype(svg_text)
    svg_text = strip_comments(svg_text)

    # Remove metadata elements
    if remove_metadata:
        svg_text = remove_metadata_elements(svg_text)

    # Remove unnecessary attributes
    svg_text = remove_unnecessary_attributes(svg_text)

    # Normalize coordinates
    svg_text = normalize_coordinates(svg_text, precision=precision)

    # Normalize whitespace
    svg_text = normalize_whitespace(svg_text)

    return svg_text


def preprocess_dataset(
    input_dir,
    output_path,
    min_length=50,
    max_length=10000,
    precision=1,
    remove_metadata=True,
):
    """
    Process all downloaded SVG JSONL files.

    Args:
        input_dir: Directory with raw JSONL files
        output_path: Path to save cleaned JSONL
        min_length: Minimum SVG character length
        max_length: Maximum SVG character length
        precision: Coordinate rounding precision
        remove_metadata: Whether to remove metadata elements
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find all JSONL files
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    stats = {
        "total_raw": 0,
        "too_short": 0,
        "too_long": 0,
        "invalid_xml": 0,
        "no_svg_root": 0,
        "kept": 0,
        "sources": {},
    }

    kept_svgs = []

    for jsonl_file in jsonl_files:
        filepath = os.path.join(input_dir, jsonl_file)
        source = jsonl_file.replace('.jsonl', '')
        source_stats = {"raw": 0, "kept": 0}

        print(f"\nProcessing {jsonl_file}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"  Cleaning {source}"):
            stats["total_raw"] += 1
            source_stats["raw"] += 1

            try:
                item = json.loads(line.strip())
                svg_text = item.get("svg", "")
            except json.JSONDecodeError:
                continue

            # Clean
            cleaned = clean_single_svg(
                svg_text, precision=precision, remove_metadata=remove_metadata
            )

            # Filter by length
            if len(cleaned) < min_length:
                stats["too_short"] += 1
                continue
            if len(cleaned) > max_length:
                stats["too_long"] += 1
                continue

            # Validate XML
            if not validate_xml(cleaned):
                stats["invalid_xml"] += 1
                continue

            # Check SVG root
            if not has_svg_root(cleaned):
                stats["no_svg_root"] += 1
                continue

            kept_svgs.append({"svg": cleaned, "source": source})
            stats["kept"] += 1
            source_stats["kept"] += 1

        stats["sources"][source] = source_stats

    # Saving cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in kept_svgs:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    # summary
    print(f"\n{'='*60}")
    print("Preprocessing Summary:")
    print(f"{'='*60}")
    print(f"Total raw SVGs:    {stats['total_raw']:,}")
    print(f"Too short (<{min_length}):  {stats['too_short']:,}")
    print(f"Too long (>{max_length}):  {stats['too_long']:,}")
    print(f"Invalid XML:       {stats['invalid_xml']:,}")
    print(f"No SVG root:       {stats['no_svg_root']:,}")
    print(f"Kept (valid):      {stats['kept']:,}")
    print(f"\nPer source:")
    for source, ss in stats["sources"].items():
        print(f"  {source}: {ss['raw']:,} raw → {ss['kept']:,} kept")

    # Saving stats
    stats_path = output_path.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default="/content/drive/MyDrive/svg-scaling/raw")
    parser.add_argument("--output_path", type=str,
                        default="/content/drive/MyDrive/svg-scaling/data/cleaned.jsonl")
    parser.add_argument("--min_length", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=10000)
    parser.add_argument("--precision", type=int, default=1)
    args = parser.parse_args()

    preprocess_dataset(
        args.input_dir, args.output_path,
        min_length=args.min_length, max_length=args.max_length,
        precision=args.precision
    )
