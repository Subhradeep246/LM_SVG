"""
SVG Validation Utilities
XML parsing, SVG structural validation, and attribute checking.
"""

from lxml import etree
import re


def check_xml_valid(svg_text):
    """Check if the SVG is valid XML."""
    try:
        etree.fromstring(svg_text.encode('utf-8'))
        return True, None
    except etree.XMLSyntaxError as e:
        return False, str(e)


def check_svg_root(svg_text):
    """Check if the SVG has a proper <svg> root element."""
    try:
        tree = etree.fromstring(svg_text.encode('utf-8'))
        tag = tree.tag
        if '}' in tag:
            tag = tag.split('}')[1]
        return tag.lower() == 'svg'
    except Exception:
        return False


def check_closed_tags(svg_text):
    """Check if all tags are properly closed."""
    try:
        etree.fromstring(svg_text.encode('utf-8'))
        return True
    except etree.XMLSyntaxError:
        return False


def check_valid_attributes(svg_text):
    """Check if SVG attributes have valid values."""
    issues = []
    try:
        tree = etree.fromstring(svg_text.encode('utf-8'))
        for elem in tree.iter():
            # Checking coordinate attributes
            for attr in ['x', 'y', 'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height',
                         'x1', 'y1', 'x2', 'y2']:
                val = elem.get(attr)
                if val is not None:
                    try:
                        float(val.replace('px', '').replace('%', ''))
                    except ValueError:
                        issues.append(f"Invalid {attr}='{val}'")

            # Checking color attributes
            for attr in ['fill', 'stroke']:
                val = elem.get(attr)
                if val is not None and val != 'none' and val != 'currentColor':
                    # Checking hex color
                    if val.startswith('#'):
                        if not re.match(r'^#[0-9a-fA-F]{3,8}$', val):
                            issues.append(f"Invalid color {attr}='{val}'")
                    # Named colors and rgb() are generally OK

            # Checking viewBox
            viewbox = elem.get('viewBox')
            if viewbox is not None:
                parts = viewbox.split()
                if len(parts) != 4:
                    issues.append(f"Invalid viewBox='{viewbox}'")

    except Exception as e:
        issues.append(f"Parse error: {e}")

    return len(issues) == 0, issues


def validate_svg_comprehensive(svg_text):
    """
    Run all validation checks on an SVG string.

    Returns:
        dict with validation results
    """
    results = {
        'xml_valid': False,
        'svg_root': False,
        'closed_tags': False,
        'valid_attributes': False,
        'xml_error': None,
        'attribute_issues': [],
    }

    # XML validity
    xml_valid, xml_error = check_xml_valid(svg_text)
    results['xml_valid'] = xml_valid
    results['xml_error'] = xml_error

    if xml_valid:
        results['svg_root'] = check_svg_root(svg_text)
        results['closed_tags'] = check_closed_tags(svg_text)
        attr_valid, issues = check_valid_attributes(svg_text)
        results['valid_attributes'] = attr_valid
        results['attribute_issues'] = issues

    # Overall structural validity
    results['structurally_valid'] = (
        results['xml_valid'] and
        results['svg_root'] and
        results['closed_tags']
    )

    return results


def validate_batch(svg_texts):
    """
    Validate a batch of SVG strings.

    Returns:
        dict with aggregate statistics
    """
    total = len(svg_texts)
    results = {
        'total': total,
        'xml_valid': 0,
        'svg_root': 0,
        'closed_tags': 0,
        'valid_attributes': 0,
        'structurally_valid': 0,
    }

    for svg in svg_texts:
        r = validate_svg_comprehensive(svg)
        for key in ['xml_valid', 'svg_root', 'closed_tags',
                     'valid_attributes', 'structurally_valid']:
            if r[key]:
                results[key] += 1

    # Compute rates
    for key in ['xml_valid', 'svg_root', 'closed_tags',
                 'valid_attributes', 'structurally_valid']:
        results[f'{key}_rate'] = results[key] / total if total > 0 else 0

    return results
