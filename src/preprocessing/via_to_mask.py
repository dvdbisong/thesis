"""
VIA JSON to Binary Mask Converter

Converts VGG Image Annotator (VIA) polygon annotations to binary mask images
for the Figshare Landsat kelp detection dataset.

Usage:
    python via_to_mask.py --input data/figshare_landsat --output data/figshare_landsat

Author: Ekaba Bisong
Date: 2026-03-19
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def load_via_annotations(json_path: Path) -> Dict:
    """
    Load VIA JSON annotation file.

    Args:
        json_path: Path to via_region_data.json

    Returns:
        Dictionary with image annotations
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_polygons(regions: List[Dict]) -> List[List[Tuple[int, int]]]:
    """
    Extract polygon coordinates from VIA regions.

    Args:
        regions: List of region dictionaries from VIA

    Returns:
        List of polygon coordinate lists
    """
    polygons = []

    for region in regions:
        shape = region.get('shape_attributes', {})

        if shape.get('name') == 'polygon':
            points_x = shape.get('all_points_x', [])
            points_y = shape.get('all_points_y', [])

            if points_x and points_y and len(points_x) == len(points_y):
                polygon = list(zip(points_x, points_y))
                polygons.append(polygon)

    return polygons


def create_binary_mask(
    image_size: Tuple[int, int],
    polygons: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """
    Create binary mask from polygon annotations.

    Args:
        image_size: (width, height) of the image
        polygons: List of polygon coordinate lists

    Returns:
        Binary mask as numpy array (H, W) with values 0 or 1
    """
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for polygon in polygons:
        if len(polygon) >= 3:  # Valid polygon needs at least 3 points
            draw.polygon(polygon, fill=1, outline=1)

    return np.array(mask, dtype=np.uint8)


def process_split(
    split_dir: Path,
    output_dir: Path,
    overwrite: bool = False
) -> Dict:
    """
    Process all images in a dataset split.

    Args:
        split_dir: Directory containing images and via_region_data.json
        output_dir: Directory to save binary masks
        overwrite: Whether to overwrite existing masks

    Returns:
        Statistics dictionary
    """
    via_path = split_dir / 'via_region_data.json'

    if not via_path.exists():
        print(f"Warning: {via_path} not found, skipping {split_dir.name}")
        return {'processed': 0, 'skipped': 0, 'errors': 0}

    # Create output masks directory
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    via_data = load_via_annotations(via_path)

    stats = {'processed': 0, 'skipped': 0, 'errors': 0, 'with_kelp': 0}

    for key, annotation in tqdm(via_data.items(), desc=f"Processing {split_dir.name}"):
        filename = annotation.get('filename', '')

        if not filename:
            continue

        # Get image path and mask output path
        image_path = split_dir / filename
        mask_name = Path(filename).stem + '_mask.png'
        mask_path = masks_dir / mask_name

        # Skip if mask exists and not overwriting
        if mask_path.exists() and not overwrite:
            stats['skipped'] += 1
            continue

        # Check if image exists
        if not image_path.exists():
            stats['errors'] += 1
            continue

        try:
            # Get image size
            with Image.open(image_path) as img:
                image_size = img.size  # (width, height)

            # Extract polygons from regions
            regions = annotation.get('regions', [])
            polygons = extract_polygons(regions)

            # Create binary mask
            mask = create_binary_mask(image_size, polygons)

            # Save mask
            mask_img = Image.fromarray(mask * 255, mode='L')  # Scale to 0-255 for visibility
            mask_img.save(mask_path)

            stats['processed'] += 1
            if np.any(mask > 0):
                stats['with_kelp'] += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            stats['errors'] += 1

    return stats


def via_json_to_masks(
    input_dir: str,
    output_dir: str = None,
    splits: List[str] = None,
    overwrite: bool = False
) -> Dict:
    """
    Convert VIA JSON annotations to binary masks for all splits.

    Args:
        input_dir: Root directory containing train/val/test splits
        output_dir: Output directory (defaults to input_dir)
        splits: List of splits to process (defaults to ['train', 'val', 'test'])
        overwrite: Whether to overwrite existing masks

    Returns:
        Statistics for all splits
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    if splits is None:
        splits = ['train', 'val', 'test']

    all_stats = {}

    for split in splits:
        split_input = input_path / split
        split_output = output_path / split

        if not split_input.exists():
            print(f"Split directory not found: {split_input}")
            continue

        print(f"\n=== Processing {split} split ===")
        stats = process_split(split_input, split_output, overwrite)
        all_stats[split] = stats

        print(f"  Processed: {stats['processed']}")
        print(f"  With kelp: {stats['with_kelp']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert VIA JSON annotations to binary masks'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing train/val/test splits'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (defaults to input directory)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to process'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing mask files'
    )

    args = parser.parse_args()

    print("VIA JSON to Binary Mask Converter")
    print("="*40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output or args.input}")
    print(f"Splits: {args.splits}")
    print(f"Overwrite: {args.overwrite}")
    print()

    stats = via_json_to_masks(
        input_dir=args.input,
        output_dir=args.output,
        splits=args.splits,
        overwrite=args.overwrite
    )

    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)

    total_processed = sum(s['processed'] for s in stats.values())
    total_with_kelp = sum(s['with_kelp'] for s in stats.values())

    print(f"Total masks created: {total_processed}")
    print(f"Images with kelp: {total_with_kelp}")
    print(f"Kelp prevalence: {total_with_kelp/total_processed*100:.1f}%" if total_processed > 0 else "N/A")


if __name__ == '__main__':
    main()
