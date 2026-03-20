"""
Tile Creator Utility for Sentinel-2 Imagery.

Standalone utility for creating tiles from preprocessed imagery.
Can be used for:
- Validating existing tiles
- Re-tiling with different parameters
- Creating tiles from stacked images

Usage:
    python -m src.preprocessing.tile_creator \
        --input path/to/stacked_12band.tif \
        --output path/to/tiles/ \
        --tile-size 512

Author: Ekaba Bisong
Date: March 2026
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


def create_tiles(
    input_path: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 0,
    min_valid_ratio: float = 0.5,
    create_masks: bool = True,
    prefix: str = "tile",
) -> Dict:
    """
    Create tiles from a multi-band raster image.

    Args:
        input_path: Path to input raster
        output_dir: Output directory for tiles
        tile_size: Size of tiles (default 512)
        overlap: Pixel overlap between tiles (default 0)
        min_valid_ratio: Minimum ratio of valid (non-zero) pixels (default 0.5)
        create_masks: Create empty mask files (default True)
        prefix: Filename prefix for tiles (default "tile")

    Returns:
        Dictionary with tiling statistics
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)

    if create_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    tile_info = []
    stride = tile_size - overlap

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        n_bands = src.count

        print(f"Input: {width}x{height}, {n_bands} bands")
        print(f"Tile size: {tile_size}x{tile_size}, overlap: {overlap}")

        # Calculate expected tiles
        n_cols = (width - tile_size) // stride + 1
        n_rows = (height - tile_size) // stride + 1
        print(f"Expected tiles: {n_rows} x {n_cols} = {n_rows * n_cols}")

        tile_idx = 0
        skipped = 0

        for row_idx, row in enumerate(range(0, height - tile_size + 1, stride)):
            for col_idx, col in enumerate(range(0, width - tile_size + 1, stride)):
                # Read tile
                window = Window(col, row, tile_size, tile_size)
                tile_data = src.read(window=window)

                # Check valid pixel ratio (using first band)
                valid_mask = tile_data[0] != 0
                valid_ratio = np.mean(valid_mask)

                if valid_ratio < min_valid_ratio:
                    skipped += 1
                    continue

                # Calculate tile transform and bounds
                tile_transform = rasterio.windows.transform(window, src.transform)
                tile_bounds = rasterio.windows.bounds(window, src.transform)

                # Save tile
                tile_filename = f"{prefix}_{tile_idx}_image.tiff"
                tile_path = images_dir / tile_filename

                tile_profile = src.profile.copy()
                tile_profile.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": tile_transform,
                })

                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(tile_data)

                # Create empty mask
                if create_masks:
                    mask_filename = f"{prefix}_{tile_idx}_mask.tiff"
                    mask_path = masks_dir / mask_filename

                    mask_profile = tile_profile.copy()
                    mask_profile.update({
                        "count": 1,
                        "dtype": "uint8",
                    })

                    with rasterio.open(mask_path, "w", **mask_profile) as dst:
                        empty_mask = np.zeros((1, tile_size, tile_size), dtype=np.uint8)
                        dst.write(empty_mask)

                # Store tile info
                tile_info.append({
                    "tile_id": tile_idx,
                    "filename": tile_filename,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "pixel_row": row,
                    "pixel_col": col,
                    "valid_ratio": float(valid_ratio),
                    "bounds": {
                        "left": tile_bounds[0],
                        "bottom": tile_bounds[1],
                        "right": tile_bounds[2],
                        "top": tile_bounds[3],
                    },
                })

                tile_idx += 1

    # Create metadata file
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "config": {
            "tile_size": tile_size,
            "overlap": overlap,
            "min_valid_ratio": min_valid_ratio,
        },
        "input_info": {
            "width": width,
            "height": height,
            "n_bands": n_bands,
            "crs": str(src.crs),
        },
        "summary": {
            "total_tiles": tile_idx,
            "skipped_tiles": skipped,
            "grid_rows": n_rows,
            "grid_cols": n_cols,
        },
        "tiles": tile_info,
    }

    metadata_path = output_path / "tile_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCreated {tile_idx} tiles, skipped {skipped} (low valid ratio)")
    print(f"Metadata saved: {metadata_path}")

    return metadata


def validate_tiles(
    tiles_dir: str,
    expected_bands: int = 12,
    expected_size: int = 512,
) -> Dict:
    """
    Validate existing tiles for consistency.

    Args:
        tiles_dir: Directory containing tiles (images/ and masks/)
        expected_bands: Expected number of bands (default 12)
        expected_size: Expected tile size (default 512)

    Returns:
        Validation report dictionary
    """
    tiles_path = Path(tiles_dir)
    images_dir = tiles_path / "images"
    masks_dir = tiles_path / "masks"

    if not images_dir.exists():
        return {"valid": False, "error": "images/ directory not found"}

    issues = []
    tile_stats = []

    # Get all image tiles
    image_files = sorted(images_dir.glob("tile_*_image.tiff"))
    print(f"Found {len(image_files)} image tiles")

    for img_path in image_files:
        tile_id = img_path.stem.replace("_image", "")

        try:
            with rasterio.open(img_path) as src:
                # Check dimensions
                if src.width != expected_size or src.height != expected_size:
                    issues.append({
                        "tile": tile_id,
                        "issue": f"Unexpected size: {src.width}x{src.height}",
                    })

                # Check bands
                if src.count != expected_bands:
                    issues.append({
                        "tile": tile_id,
                        "issue": f"Unexpected bands: {src.count}",
                    })

                # Check for all-zero tiles
                data = src.read()
                valid_ratio = np.mean(data[0] != 0)

                tile_stats.append({
                    "tile_id": tile_id,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "valid_ratio": float(valid_ratio),
                    "dtype": str(src.dtypes[0]),
                })

                if valid_ratio == 0:
                    issues.append({
                        "tile": tile_id,
                        "issue": "All-zero tile",
                    })

        except Exception as e:
            issues.append({
                "tile": tile_id,
                "issue": f"Read error: {e}",
            })

    # Check masks
    if masks_dir.exists():
        mask_files = sorted(masks_dir.glob("tile_*_mask.tiff"))
        if len(mask_files) != len(image_files):
            issues.append({
                "tile": "all",
                "issue": f"Mask count mismatch: {len(mask_files)} masks for {len(image_files)} images",
            })

    report = {
        "valid": len(issues) == 0,
        "tiles_dir": str(tiles_dir),
        "total_tiles": len(image_files),
        "issues_count": len(issues),
        "issues": issues,
        "stats": {
            "min_valid_ratio": min(t["valid_ratio"] for t in tile_stats) if tile_stats else 0,
            "max_valid_ratio": max(t["valid_ratio"] for t in tile_stats) if tile_stats else 0,
            "mean_valid_ratio": np.mean([t["valid_ratio"] for t in tile_stats]) if tile_stats else 0,
        },
    }

    print(f"\nValidation {'PASSED' if report['valid'] else 'FAILED'}")
    print(f"Total tiles: {report['total_tiles']}")
    print(f"Issues found: {report['issues_count']}")

    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue['tile']}: {issue['issue']}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tile creator and validator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create tiles from image")
    create_parser.add_argument("--input", required=True, help="Input raster path")
    create_parser.add_argument("--output", required=True, help="Output directory")
    create_parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    create_parser.add_argument("--overlap", type=int, default=0, help="Tile overlap")
    create_parser.add_argument("--min-valid", type=float, default=0.5, help="Min valid ratio")
    create_parser.add_argument("--no-masks", action="store_true", help="Skip mask creation")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate existing tiles")
    validate_parser.add_argument("--dir", required=True, help="Tiles directory")
    validate_parser.add_argument("--bands", type=int, default=12, help="Expected bands")
    validate_parser.add_argument("--size", type=int, default=512, help="Expected tile size")

    args = parser.parse_args()

    if args.command == "create":
        create_tiles(
            input_path=args.input,
            output_dir=args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_valid_ratio=args.min_valid,
            create_masks=not args.no_masks,
        )
    elif args.command == "validate":
        report = validate_tiles(
            tiles_dir=args.dir,
            expected_bands=args.bands,
            expected_size=args.size,
        )
        # Save report
        report_path = Path(args.dir) / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {report_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
