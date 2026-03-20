"""
Preprocessing Pipeline for Multi-Temporal Sentinel-2 Imagery.

This script processes raw GEE downloads to match Mohsen's data format:
- Resamples 20m bands to 10m
- Aligns auxiliary data (Bathymetry, Substrate)
- Stacks all 12 bands
- Creates 512x512 tiles

Usage:
    python -m src.preprocessing.preprocess_multitemporal \
        --input data/bc_sentinel2_multitemporal/raw/ \
        --output data/bc_sentinel2_multitemporal/Tiles/ \
        --auxiliary data/bc_sentinel2/new/Masks\ 10\ scenes/

Author: Ekaba Bisong
Date: March 2026
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window

# Tile configuration (matching Mohsen's format)
TILE_SIZE = 512
MIN_VALID_RATIO = 0.5  # Minimum non-zero pixel ratio to keep a tile

# Band order for 12-band stack
BAND_ORDER = [
    "B2", "B3", "B4", "B8",           # 10m bands (indices 0-3)
    "B5", "B6", "B7", "B8A", "B11", "B12",  # 20m bands (indices 4-9)
    "Substrate", "Bathymetry"          # Auxiliary (indices 10-11)
]

# Mapping from site codes to existing auxiliary data paths
# These are static (don't change over time) so we reuse from labeled data
AUXILIARY_SITE_MAPPING = {
    "T09UXQ": "20210727T191911_20210727T192721_T09UXQ",
    "T09UYQ": "20220806T191919_20220806T192707_T09UYQ",
    "T10UCU": "20220806T191919_20220806T192707_T10UCU",
    "T09UXS": "20230804T192909_20230804T192942_T09UXS",
    "T10UCA": "20230816T191911_20230816T192348_T10UCA",
    "T09UWT": "20230819T192911_20230819T193100_T09UWT",
    "T09UUU": "20230902T195919_20230902T195917_T09UUU",
    "T09UWS": "20200908T192939_20200908T193355_T09UWS",
    "T09UXR": "20200908T192939_20200908T193355_T09UXR",
    "T10UDU": None,  # May not have auxiliary data
}


@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    image_id: str
    site: str
    n_tiles: int
    tiles_dir: str
    success: bool
    error: Optional[str] = None


class Sentinel2Preprocessor:
    """
    Preprocessor to match Mohsen's data format.

    Input: Raw Sentinel-2 bands from GEE
    Output: 12-band tiles (512x512) with auxiliary layers
    """

    def __init__(self, auxiliary_dir: str, tile_size: int = TILE_SIZE):
        """
        Args:
            auxiliary_dir: Path to existing auxiliary data (Bathymetry, Substrate)
            tile_size: Size of output tiles (default 512)
        """
        self.auxiliary_dir = Path(auxiliary_dir)
        self.tile_size = tile_size

    def find_auxiliary_for_site(self, site_id: str) -> Optional[Path]:
        """
        Find auxiliary data directory for a given site.

        Args:
            site_id: MGRS tile code (e.g., "T09UXQ")

        Returns:
            Path to auxiliary directory or None if not found
        """
        # Check mapping first
        if site_id in AUXILIARY_SITE_MAPPING:
            scene_id = AUXILIARY_SITE_MAPPING[site_id]
            if scene_id is None:
                return None
            aux_path = self.auxiliary_dir / scene_id
            if aux_path.exists():
                return aux_path

        # Fall back to searching by site code
        for subdir in self.auxiliary_dir.iterdir():
            if subdir.is_dir() and site_id in subdir.name:
                return subdir

        return None

    def resample_to_reference(
        self,
        src_path: str,
        reference_path: str,
        output_path: str,
        resampling_method: Resampling = Resampling.bilinear
    ) -> bool:
        """
        Resample a raster to match a reference raster's grid.

        Args:
            src_path: Path to source raster
            reference_path: Path to reference raster (defines target grid)
            output_path: Path for output raster
            resampling_method: Resampling algorithm

        Returns:
            True if successful, False otherwise
        """
        try:
            with rasterio.open(reference_path) as ref:
                ref_crs = ref.crs
                ref_transform = ref.transform
                ref_width = ref.width
                ref_height = ref.height
                ref_bounds = ref.bounds

            with rasterio.open(src_path) as src:
                # Prepare output array
                out_data = np.zeros((src.count, ref_height, ref_width), dtype=src.dtypes[0])

                # Reproject each band
                for i in range(src.count):
                    reproject(
                        source=rasterio.band(src, i + 1),
                        destination=out_data[i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=resampling_method,
                    )

                # Write output
                profile = {
                    "driver": "GTiff",
                    "height": ref_height,
                    "width": ref_width,
                    "count": src.count,
                    "dtype": out_data.dtype,
                    "crs": ref_crs,
                    "transform": ref_transform,
                    "compress": "lzw",
                }

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(out_data)

            return True

        except Exception as e:
            print(f"Error resampling {src_path}: {e}")
            return False

    def stack_bands(
        self,
        bands_10m_path: str,
        bands_20m_path: str,
        substrate_path: Optional[str],
        bathymetry_path: Optional[str],
        output_path: str,
    ) -> bool:
        """
        Stack all 12 bands into a single GeoTIFF.

        Band order:
        1-4: B2, B3, B4, B8 (10m)
        5-10: B5, B6, B7, B8A, B11, B12 (resampled to 10m)
        11: Substrate
        12: Bathymetry

        Args:
            bands_10m_path: Path to 10m bands (4 bands)
            bands_20m_path: Path to 20m bands resampled to 10m (6 bands)
            substrate_path: Path to substrate raster (optional)
            bathymetry_path: Path to bathymetry raster (optional)
            output_path: Output path for stacked raster

        Returns:
            True if successful
        """
        try:
            # Read 10m bands
            with rasterio.open(bands_10m_path) as src_10m:
                profile = src_10m.profile.copy()
                data_10m = src_10m.read()  # Shape: (4, H, W)
                height, width = data_10m.shape[1], data_10m.shape[2]

            # Read 20m bands (already resampled)
            with rasterio.open(bands_20m_path) as src_20m:
                data_20m = src_20m.read()  # Shape: (6, H, W)

            # Initialize auxiliary arrays
            data_substrate = np.zeros((height, width), dtype=np.float32)
            data_bathymetry = np.zeros((height, width), dtype=np.float32)

            # Read substrate if available
            if substrate_path and Path(substrate_path).exists():
                with rasterio.open(substrate_path) as src:
                    data_substrate = src.read(1).astype(np.float32)

            # Read bathymetry if available
            if bathymetry_path and Path(bathymetry_path).exists():
                with rasterio.open(bathymetry_path) as src:
                    data_bathymetry = src.read(1).astype(np.float32)

            # Stack all bands
            stacked = np.concatenate([
                data_10m.astype(np.float32),           # Bands 1-4
                data_20m.astype(np.float32),           # Bands 5-10
                data_substrate[np.newaxis, :, :],      # Band 11
                data_bathymetry[np.newaxis, :, :],     # Band 12
            ], axis=0)

            # Update profile for 12 bands
            profile.update({
                "count": 12,
                "dtype": "float32",
                "compress": "lzw",
            })

            # Write stacked image
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(stacked)

            return True

        except Exception as e:
            print(f"Error stacking bands: {e}")
            return False

    def create_tiles(
        self,
        stacked_image_path: str,
        output_dir: str,
        overlap: int = 0,
        create_empty_masks: bool = True,
    ) -> List[str]:
        """
        Tile the full image into 512x512 chips.

        Args:
            stacked_image_path: Path to 12-band stacked image
            output_dir: Directory for output tiles
            overlap: Pixel overlap between tiles (default 0)
            create_empty_masks: Create empty mask files (for unlabeled data)

        Returns:
            List of tile paths
        """
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        masks_dir = output_path / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = []

        with rasterio.open(stacked_image_path) as src:
            height, width = src.height, src.width
            stride = self.tile_size - overlap

            tile_idx = 0
            for row in range(0, height - self.tile_size + 1, stride):
                for col in range(0, width - self.tile_size + 1, stride):
                    # Read tile
                    window = Window(col, row, self.tile_size, self.tile_size)
                    tile_data = src.read(window=window)

                    # Check valid pixel ratio (using first band)
                    valid_mask = tile_data[0] != 0
                    valid_ratio = np.mean(valid_mask)

                    if valid_ratio < MIN_VALID_RATIO:
                        continue

                    # Calculate tile transform
                    tile_transform = rasterio.windows.transform(window, src.transform)

                    # Save tile
                    tile_path = images_dir / f"tile_{tile_idx}_image.tiff"
                    tile_profile = src.profile.copy()
                    tile_profile.update({
                        "height": self.tile_size,
                        "width": self.tile_size,
                        "transform": tile_transform,
                    })

                    with rasterio.open(tile_path, "w", **tile_profile) as dst:
                        dst.write(tile_data)

                    # Create empty mask (no labels for multi-temporal data)
                    if create_empty_masks:
                        mask_path = masks_dir / f"tile_{tile_idx}_mask.tiff"
                        mask_profile = tile_profile.copy()
                        mask_profile.update({
                            "count": 1,
                            "dtype": "uint8",
                        })

                        with rasterio.open(mask_path, "w", **mask_profile) as dst:
                            empty_mask = np.zeros(
                                (1, self.tile_size, self.tile_size),
                                dtype=np.uint8
                            )
                            dst.write(empty_mask)

                    tile_paths.append(str(tile_path))
                    tile_idx += 1

        return tile_paths

    def process_image(
        self,
        bands_10m_path: str,
        bands_20m_path: str,
        site_id: str,
        image_id: str,
        output_base: str,
        temp_dir: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Full preprocessing pipeline for a single image.

        Args:
            bands_10m_path: Path to 10m bands GeoTIFF
            bands_20m_path: Path to 20m bands GeoTIFF
            site_id: MGRS tile code
            image_id: Unique image identifier
            output_base: Base output directory for tiles
            temp_dir: Temporary directory for intermediate files

        Returns:
            ProcessingResult with status and tile count
        """
        try:
            # Create temporary directory
            if temp_dir is None:
                temp_dir = Path(output_base) / "temp"
            temp_path = Path(temp_dir)
            temp_path.mkdir(parents=True, exist_ok=True)

            print(f"  Processing {image_id}...")

            # Step 1: Resample 20m bands to match 10m bands
            bands_20m_resampled = temp_path / f"{image_id}_20m_resampled.tif"
            print(f"    Resampling 20m bands to 10m...")

            success = self.resample_to_reference(
                bands_20m_path,
                bands_10m_path,
                str(bands_20m_resampled),
            )
            if not success:
                return ProcessingResult(
                    image_id=image_id,
                    site=site_id,
                    n_tiles=0,
                    tiles_dir="",
                    success=False,
                    error="Failed to resample 20m bands",
                )

            # Step 2: Find and align auxiliary data
            aux_dir = self.find_auxiliary_for_site(site_id)
            substrate_aligned = None
            bathymetry_aligned = None

            if aux_dir:
                print(f"    Aligning auxiliary data from {aux_dir.name}...")

                # Find auxiliary files
                substrate_files = list(aux_dir.glob("*Substrate*.tif"))
                bathymetry_files = list(aux_dir.glob("*Bathymetry*.tif"))

                if substrate_files:
                    substrate_aligned = temp_path / f"{image_id}_substrate_aligned.tif"
                    self.resample_to_reference(
                        str(substrate_files[0]),
                        bands_10m_path,
                        str(substrate_aligned),
                    )

                if bathymetry_files:
                    bathymetry_aligned = temp_path / f"{image_id}_bathymetry_aligned.tif"
                    self.resample_to_reference(
                        str(bathymetry_files[0]),
                        bands_10m_path,
                        str(bathymetry_aligned),
                    )
            else:
                print(f"    Warning: No auxiliary data found for site {site_id}")

            # Step 3: Stack all bands
            stacked_path = temp_path / f"{image_id}_stacked_12band.tif"
            print(f"    Stacking 12 bands...")

            success = self.stack_bands(
                bands_10m_path,
                str(bands_20m_resampled),
                str(substrate_aligned) if substrate_aligned else None,
                str(bathymetry_aligned) if bathymetry_aligned else None,
                str(stacked_path),
            )
            if not success:
                return ProcessingResult(
                    image_id=image_id,
                    site=site_id,
                    n_tiles=0,
                    tiles_dir="",
                    success=False,
                    error="Failed to stack bands",
                )

            # Step 4: Create tiles
            tiles_dir = Path(output_base) / f"{image_id}_{site_id}"
            print(f"    Creating {self.tile_size}x{self.tile_size} tiles...")

            tile_paths = self.create_tiles(str(stacked_path), str(tiles_dir))

            print(f"    Created {len(tile_paths)} tiles")

            # Clean up temp files
            for f in [bands_20m_resampled, substrate_aligned, bathymetry_aligned, stacked_path]:
                if f and Path(f).exists():
                    Path(f).unlink()

            return ProcessingResult(
                image_id=image_id,
                site=site_id,
                n_tiles=len(tile_paths),
                tiles_dir=str(tiles_dir),
                success=True,
            )

        except Exception as e:
            return ProcessingResult(
                image_id=image_id,
                site=site_id,
                n_tiles=0,
                tiles_dir="",
                success=False,
                error=str(e),
            )


def process_from_manifest(
    manifest_path: str,
    raw_dir: str,
    output_dir: str,
    auxiliary_dir: str,
) -> Dict:
    """
    Process all images listed in a download manifest.

    Args:
        manifest_path: Path to download_manifest.json
        raw_dir: Directory containing raw GEE downloads
        output_dir: Output directory for tiles
        auxiliary_dir: Directory with auxiliary data

    Returns:
        Processing summary dictionary
    """
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Initialize preprocessor
    preprocessor = Sentinel2Preprocessor(auxiliary_dir)

    # Process each image
    results = []
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_info in manifest["images"]:
        image_id = image_info["image_id"]
        site = image_info["site"]
        files = image_info.get("files", {})

        # Find input files
        bands_10m = raw_path / files.get("10m_bands", f"{image_id}_B2B3B4B8.tif")
        bands_20m = raw_path / files.get("20m_bands", f"{image_id}_B5B6B7B8A_B11B12.tif")

        if not bands_10m.exists():
            print(f"Skipping {image_id}: 10m bands not found at {bands_10m}")
            results.append({
                "image_id": image_id,
                "site": site,
                "success": False,
                "error": "10m bands not found",
            })
            continue

        if not bands_20m.exists():
            print(f"Skipping {image_id}: 20m bands not found at {bands_20m}")
            results.append({
                "image_id": image_id,
                "site": site,
                "success": False,
                "error": "20m bands not found",
            })
            continue

        # Process image
        result = preprocessor.process_image(
            bands_10m_path=str(bands_10m),
            bands_20m_path=str(bands_20m),
            site_id=site,
            image_id=image_id,
            output_base=str(output_path),
        )

        results.append({
            "image_id": result.image_id,
            "site": result.site,
            "n_tiles": result.n_tiles,
            "tiles_dir": result.tiles_dir,
            "success": result.success,
            "error": result.error,
        })

    # Create processing log
    log = {
        "timestamp": datetime.now().isoformat(),
        "manifest_path": manifest_path,
        "raw_dir": raw_dir,
        "output_dir": output_dir,
        "auxiliary_dir": auxiliary_dir,
        "summary": {
            "total_images": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_tiles": sum(r.get("n_tiles", 0) for r in results),
        },
        "results": results,
    }

    # Save log
    log_path = output_path / "preprocessing_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print("Preprocessing Summary")
    print(f"{'='*60}")
    print(f"Total images: {log['summary']['total_images']}")
    print(f"Successful: {log['summary']['successful']}")
    print(f"Failed: {log['summary']['failed']}")
    print(f"Total tiles created: {log['summary']['total_tiles']}")
    print(f"Log saved: {log_path}")

    return log


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess multi-temporal Sentinel-2 imagery"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing raw GEE downloads",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tiles",
    )
    parser.add_argument(
        "--auxiliary",
        type=str,
        required=True,
        help="Directory with auxiliary data (Bathymetry, Substrate)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to download manifest (default: input/download_manifest.json)",
    )

    args = parser.parse_args()

    # Find manifest
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = Path(args.input) / "download_manifest.json"

    if not Path(manifest_path).exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run gee_download.py first to create the manifest.")
        return

    # Run preprocessing
    process_from_manifest(
        manifest_path=str(manifest_path),
        raw_dir=args.input,
        output_dir=args.output,
        auxiliary_dir=args.auxiliary,
    )


if __name__ == "__main__":
    main()
