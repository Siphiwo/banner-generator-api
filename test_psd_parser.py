"""
Simple test script for PSD parser functionality.

This script demonstrates how to use the PSD parser and validates
the layer classification logic.
"""

from app.services.psd_parser import PSDParser, LayerConstraints


def test_layer_constraints():
    """Test layer constraint parsing from names."""
    print("\n=== Testing Layer Constraints ===")
    
    # Test lock modifier
    c1 = LayerConstraints("logo:brand [lock]")
    assert c1.lock is True
    print("✓ Lock modifier detected")
    
    # Test anchor modifier
    c2 = LayerConstraints("text:headline [anchor:top-left]")
    assert c2.anchor == "top-left"
    print("✓ Anchor modifier detected")
    
    # Test scale modifiers
    c3 = LayerConstraints("product:shoe [max-scale=1.5] [min-scale=0.8]")
    assert c3.max_scale == 1.5
    assert c3.min_scale == 0.8
    print("✓ Scale modifiers detected")
    
    # Test flex modifier
    c4 = LayerConstraints("decor:pattern [flex]")
    assert c4.flex is True
    print("✓ Flex modifier detected")
    
    print("All constraint tests passed!\n")


def test_psd_parsing(psd_path: str):
    """Test PSD file parsing."""
    print(f"\n=== Testing PSD Parsing: {psd_path} ===")
    
    parser = PSDParser(psd_path)
    
    if not parser.parse():
        print("✗ Failed to parse PSD file")
        return False
    
    print(f"✓ PSD parsed successfully")
    print(f"  Dimensions: {parser.get_dimensions()}")
    print(f"  Total layers: {len(parser.semantic_layers)}")
    
    # Show warnings
    if parser.warnings:
        print(f"\n  Warnings:")
        for warning in parser.warnings:
            print(f"    - {warning}")
    
    # Show background layer
    if parser.background_layer:
        print(f"\n  Background layer: {parser.background_layer.name}")
        print(f"    BBox: {parser.background_layer.bbox}")
    
    # Show semantic breakdown
    role_counts = {}
    for layer in parser.semantic_layers:
        role_counts[layer.role] = role_counts.get(layer.role, 0) + 1
    
    print(f"\n  Semantic breakdown:")
    for role, count in sorted(role_counts.items()):
        print(f"    {role}: {count}")
    
    # Test region extraction
    text_regions = parser.get_regions_by_role("text")
    logo_regions = parser.get_regions_by_role("logo")
    protected_regions = parser.get_protected_regions()
    
    print(f"\n  Extracted regions:")
    print(f"    Text: {len(text_regions)}")
    print(f"    Logo: {len(logo_regions)}")
    print(f"    Protected (total): {len(protected_regions)}")
    
    # Test flattened export
    output_path = psd_path.replace(".psd", "_test_flattened.png")
    if parser.export_flattened_image(output_path):
        print(f"\n✓ Flattened image exported to: {output_path}")
    else:
        print(f"\n✗ Failed to export flattened image")
    
    return True


if __name__ == "__main__":
    # Test constraint parsing
    test_layer_constraints()
    
    # Test PSD parsing (if a PSD file is provided)
    import sys
    if len(sys.argv) > 1:
        psd_path = sys.argv[1]
        test_psd_parsing(psd_path)
    else:
        print("\nTo test PSD parsing, run:")
        print("  python test_psd_parser.py path/to/your/file.psd")
