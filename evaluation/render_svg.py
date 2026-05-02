"""
SVG Rendering Utilities
Render SVG strings to PNG using CairoSVG, create visualization grids.
"""

import os
import numpy as np


def render_svg_to_png(svg_text, output_path=None, width=200, height=200):
    """
    Render SVG string to PNG.

    Args:
        svg_text: SVG code as string
        output_path: Path to save PNG (None = return bytes)
        width: Output width
        height: Output height

    Returns:
        True if successful, False otherwise
    """
    # CairoSVG STRICTLY requires the xmlns attribute. 
    # Since we stripped it during preprocessing to save tokens, we must inject it back!
    if '<svg' in svg_text and 'xmlns=' not in svg_text:
        svg_text = svg_text.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        
    try:
        import cairosvg
        if output_path:
            cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                write_to=output_path,
                output_width=width,
                output_height=height,
            )
        else:
            return cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=width,
                output_height=height,
            )
        return True
    except Exception as e:
        return False


def render_batch(svg_texts, output_dir, prefix='sample', width=200, height=200):
    """
    Render a batch of SVGs to PNG files.

    Returns:
        list of (index, success, path) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, svg in enumerate(svg_texts):
        path = os.path.join(output_dir, f'{prefix}_{i:03d}.png')
        success = render_svg_to_png(svg, path, width, height)
        results.append((i, bool(success), path))

    success_count = sum(1 for _, s, _ in results if s)
    print(f"Rendered {success_count}/{len(svg_texts)} SVGs successfully")

    return results


def create_sample_grid(svg_texts, output_path, cols=5, cell_size=200, labels=None):
    """
    Create a grid of rendered SVG samples.

    Args:
        svg_texts: List of SVG strings
        output_path: Path to save grid image
        cols: Number of columns
        cell_size: Size of each cell in pixels
        labels: Optional labels for each cell
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
    except ImportError:
        print("PIL not installed. Skipping grid creation.")
        return

    n = len(svg_texts)
    rows = (n + cols - 1) // cols
    label_height = 30 if labels else 0

    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_height)
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    for i, svg in enumerate(svg_texts):
        row = i // cols
        col = i % cols
        x = col * cell_size
        y = row * (cell_size + label_height)

        # Render SVG
        try:
            png_bytes = render_svg_to_png(svg, width=cell_size, height=cell_size)
            if png_bytes and png_bytes is not True:
                img = Image.open(io.BytesIO(png_bytes))
                if img.mode == 'RGBA':
                    grid.paste(img, (x, y), img)
                else:
                    grid.paste(img, (x, y))
            else:
                # Draw placeholder
                draw.rectangle([x, y, x + cell_size, y + cell_size],
                              outline='#ccc', width=2)
                draw.text((x + 10, y + cell_size // 2), 'Render\nfailed',
                         fill='#999')
        except Exception:
            draw.rectangle([x, y, x + cell_size, y + cell_size],
                          outline='#ccc', width=2)
            draw.text((x + 10, y + cell_size // 2), 'Error', fill='#999')

        # Label
        if labels and i < len(labels):
            draw.text((x + 5, y + cell_size + 2), str(labels[i]),
                     fill='#333')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    print(f"Saved sample grid: {output_path}")


def create_prefix_comparison(prefixes, completions, output_path, cell_size=200):
    """
    Create side-by-side comparison of prefix → completion → rendered result.

    Args:
        prefixes: List of SVG prefix strings
        completions: List of completed SVG strings
        output_path: Path to save comparison image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
    except ImportError:
        print("PIL not installed.")
        return

    n = len(prefixes)
    col_width = cell_size
    row_height = cell_size + 40  # Extra space for labels
    grid_w = 3 * col_width  # prefix | completion | rendered
    grid_h = n * row_height + 40  # Header

    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    # Headers
    headers = ['Prefix', 'Completion', 'Rendered']
    for j, header in enumerate(headers):
        draw.text((j * col_width + 10, 10), header, fill='#333')

    for i in range(n):
        y = 40 + i * row_height

        # Prefix rendering
        try:
            # Try to render prefix as SVG (may not be complete)
            prefix_svg = prefixes[i]
            if not prefix_svg.strip().endswith('</svg>'):
                prefix_svg += '</svg>'
            png_bytes = render_svg_to_png(prefix_svg, width=col_width, height=cell_size)
            if png_bytes and png_bytes is not True:
                img = Image.open(io.BytesIO(png_bytes))
                if img.mode == 'RGBA':
                    grid.paste(img, (0, y), img)
                else:
                    grid.paste(img, (0, y))
            else:
                draw.text((10, y + 10), 'Prefix\n(partial)', fill='#999')
        except Exception:
            draw.text((10, y + 10), 'Prefix\n(partial)', fill='#999')

        # Completion text preview
        comp_text = completions[i][:100] + '...' if len(completions[i]) > 100 else completions[i]
        # Wrap text
        lines = [comp_text[j:j+25] for j in range(0, len(comp_text), 25)]
        for k, line in enumerate(lines[:8]):
            draw.text((col_width + 5, y + k * 15), line, fill='#333')

        # Rendered completion
        try:
            comp_svg = completions[i]
            if '<svg' in comp_svg and not comp_svg.strip().endswith('</svg>'):
                # Try to aggressively close any open tags to force rendering
                if comp_svg.rfind('<') > comp_svg.rfind('>'):
                    comp_svg = comp_svg[:comp_svg.rfind('<')]
                comp_svg += '</svg>'
            png_bytes = render_svg_to_png(comp_svg, width=col_width, height=cell_size)
            if png_bytes and png_bytes is not True:
                img = Image.open(io.BytesIO(png_bytes))
                if img.mode == 'RGBA':
                    grid.paste(img, (2 * col_width, y), img)
                else:
                    grid.paste(img, (2 * col_width, y))
            else:
                draw.text((2 * col_width + 10, y + 10), 'Render\nfailed', fill='#999')
        except Exception:
            draw.text((2 * col_width + 10, y + 10), 'Render\nfailed', fill='#999')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    print(f"Saved prefix comparison: {output_path}")
