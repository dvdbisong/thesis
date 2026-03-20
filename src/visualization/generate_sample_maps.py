"""
Generate 4K Publication-Ready Sample Maps

Creates high-resolution RGB composites with kelp mask overlays for each BC site.
Output: 400 DPI PNG and PDF files with cartographic elements.

Usage:
    python generate_sample_maps.py --output results/raw/phase_0_eda/sample_maps/

Author: Ekaba Bisong
Date: 2026-03-19
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from matplotlib_scalebar.scalebar import ScaleBar  # Commented out, not used
import rasterio
from tqdm import tqdm


# Configuration
DPI = 400
FIGSIZE = (9.6, 5.4)  # Results in ~3840x2160 at 400 DPI


def load_tile_and_mask(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a tile and its corresponding mask.

    Returns:
        Tuple of (rgb_array, mask_array, metadata)
    """
    with rasterio.open(image_path) as src:
        # Read bands 4, 3, 2 (indices 2, 1, 0) for RGB
        rgb = np.stack([
            src.read(3),  # B4 - Red
            src.read(2),  # B3 - Green
            src.read(1),  # B2 - Blue
        ], axis=-1)

        meta = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'res': src.res
        }

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    return rgb, mask, meta


def normalize_rgb(rgb: np.ndarray, percentile: float = 98) -> np.ndarray:
    """Normalize RGB for visualization using percentile stretching."""
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[:, :, i]
        valid = band[band > 0]

        if len(valid) > 0:
            p_low = np.percentile(valid, 100 - percentile)
            p_high = np.percentile(valid, percentile)
            rgb_norm[:, :, i] = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return rgb_norm


def create_mosaic(tiles_dir: Path, n_tiles: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mosaic from multiple tiles for a scene.

    Args:
        tiles_dir: Directory containing images/ and masks/ subdirectories
        n_tiles: Number of tiles to include (2x2 grid)

    Returns:
        Tuple of (rgb_mosaic, mask_mosaic)
    """
    images_dir = tiles_dir / 'images'
    masks_dir = tiles_dir / 'masks'

    tile_files = sorted(images_dir.glob('*.tiff'))[:n_tiles]

    if len(tile_files) == 0:
        raise ValueError(f"No tiles found in {images_dir}")

    # Load first tile to get dimensions
    rgb0, mask0, _ = load_tile_and_mask(
        tile_files[0],
        masks_dir / tile_files[0].name.replace('_image.tiff', '_mask.tiff')
    )

    h, w = rgb0.shape[:2]
    grid_size = int(np.ceil(np.sqrt(n_tiles)))

    # Create mosaic arrays
    mosaic_rgb = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.float32)
    mosaic_mask = np.zeros((h * grid_size, w * grid_size), dtype=np.uint8)

    for idx, tile_path in enumerate(tile_files):
        row = idx // grid_size
        col = idx % grid_size

        mask_path = masks_dir / tile_path.name.replace('_image.tiff', '_mask.tiff')

        if not mask_path.exists():
            continue

        rgb, mask, _ = load_tile_and_mask(tile_path, mask_path)
        rgb_norm = normalize_rgb(rgb)

        y_start = row * h
        y_end = (row + 1) * h
        x_start = col * w
        x_end = (col + 1) * w

        mosaic_rgb[y_start:y_end, x_start:x_end] = rgb_norm
        mosaic_mask[y_start:y_end, x_start:x_end] = mask

    return mosaic_rgb, mosaic_mask


def generate_sample_map(
    scene_dir: Path,
    output_dir: Path,
    dpi: int = DPI,
    n_tiles: int = 4
):
    """
    Generate a publication-ready sample map for a scene.

    Args:
        scene_dir: Directory containing the scene tiles
        output_dir: Output directory for the map
        dpi: Output resolution (dots per inch)
        n_tiles: Number of tiles to mosaic
    """
    site_code = scene_dir.name.split('_')[-1]
    scene_date = scene_dir.name.split('T')[0]

    # Create mosaic
    try:
        rgb_mosaic, mask_mosaic = create_mosaic(scene_dir, n_tiles)
    except Exception as e:
        print(f"  Error creating mosaic for {site_code}: {e}")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=dpi)

    # Left: RGB composite
    axes[0].imshow(rgb_mosaic)
    axes[0].set_title(f'RGB Composite', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Add scale bar (assuming 10m resolution, 512px = 5120m)
    # scalebar = ScaleBar(10, 'm', location='lower right', box_alpha=0.7)
    # axes[0].add_artist(scalebar)

    # Right: RGB with kelp overlay (direct RGB modification for visibility)
    rgb_with_overlay = rgb_mosaic.copy()
    kelp_pixels = mask_mosaic == 1
    rgb_with_overlay[kelp_pixels, 0] = 1.0  # Red channel
    rgb_with_overlay[kelp_pixels, 1] = 0.0  # Green channel
    rgb_with_overlay[kelp_pixels, 2] = 0.0  # Blue channel
    axes[1].imshow(rgb_with_overlay)

    axes[1].set_title('RGB + Kelp Mask Overlay', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    # Add legend
    kelp_patch = mpatches.Patch(color='red', alpha=0.6, label='Kelp')
    axes[1].legend(handles=[kelp_patch], loc='upper right', fontsize=8)

    # Calculate kelp statistics
    kelp_pct = (np.sum(mask_mosaic > 0) / mask_mosaic.size) * 100

    # Main title with metadata
    fig.suptitle(
        f'Site: {site_code}  |  Date: {scene_date[:4]}-{scene_date[4:6]}-{scene_date[6:8]}  |  '
        f'Kelp Coverage: {kelp_pct:.2f}%',
        fontsize=11, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # PNG
    png_path = output_dir / f'{site_code}_sample_map.png'
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    # PDF
    pdf_path = output_dir / f'{site_code}_sample_map.pdf'
    plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    plt.close()

    print(f"  Saved: {png_path.name}, {pdf_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate 4K publication-ready sample maps'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/ebisong/Documents/code/uvic/thesis/data/bc_sentinel2/new/Tiles 10 Scenes',
        help='Directory containing scene tiles'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/Users/ebisong/Documents/code/uvic/thesis/results/raw/phase_0_eda/sample_maps',
        help='Output directory for sample maps'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=DPI,
        help=f'Output DPI (default: {DPI})'
    )
    parser.add_argument(
        '--n-tiles',
        type=int,
        default=4,
        help='Number of tiles per mosaic'
    )
    parser.add_argument(
        '--sites',
        type=str,
        nargs='+',
        default=None,
        help='Specific sites to process (default: all)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    print("="*60)
    print("4K Sample Map Generator")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print()

    # Process each scene
    scenes = sorted(data_dir.iterdir())

    for scene_dir in tqdm(scenes, desc="Generating maps"):
        if not scene_dir.is_dir():
            continue

        site_code = scene_dir.name.split('_')[-1]

        # Filter by sites if specified
        if args.sites and site_code not in args.sites:
            continue

        print(f"\nProcessing {site_code}...")
        generate_sample_map(
            scene_dir,
            output_dir,
            dpi=args.dpi,
            n_tiles=args.n_tiles
        )

    print()
    print("="*60)
    print(f"Maps saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
