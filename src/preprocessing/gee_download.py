"""
Google Earth Engine Download Script for Multi-Temporal Sentinel-2 Imagery.

This script downloads Sentinel-2 L2A imagery for BC kelp monitoring sites
across multiple seasons and years to enable empirical testing of the
dual-layer MSE/PSE framework.

Usage:
    python -m src.preprocessing.gee_download --output data/bc_sentinel2_multitemporal/raw/

Requirements:
    - Google Earth Engine account and authentication
    - earthengine-api package
    - geemap package (optional, for visualization)

Author: Ekaba Bisong
Date: March 2026
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee

# Site definitions from site_locations.json
# Coordinates are center points; we'll create bounding boxes around them
SITES = {
    "T09UUU": {
        "center": (52.9675, -131.6354),
        "utm_zone": "9N",
        "location_name": "Haida Gwaii (Southern)",
    },
    "T09UWT": {
        "center": (51.9352, -128.4793),
        "utm_zone": "9N",
        "location_name": "Central Coast (Bella Bella)",
    },
    "T09UWS": {
        "center": (50.8216, -127.6195),
        "utm_zone": "9N",
        "location_name": "North Vancouver Island (Port Hardy)",
    },
    "T09UXS": {
        "center": (50.5347, -126.9028),
        "utm_zone": "9N",
        "location_name": "Johnstone Strait",
    },
    "T10UCA": {
        "center": (50.2282, -125.4096),
        "utm_zone": "10N",
        "location_name": "Discovery Islands (Campbell River)",
    },
    "T09UXR": {
        "center": (49.8216, -127.0050),
        "utm_zone": "9N",
        "location_name": "Nootka Sound",
    },
    "T09UXQ": {
        "center": (49.3962, -126.5291),
        "utm_zone": "9N",
        "location_name": "Clayoquot Sound (Tofino)",
    },
    "T09UYQ": {
        "center": (48.7577, -125.0949),
        "utm_zone": "9N",
        "location_name": "Barkley Sound (Ucluelet)",
    },
    "T10UCU": {
        "center": (48.5808, -124.6013),
        "utm_zone": "10N",
        "location_name": "Southern Vancouver Island (Sooke)",
    },
    "T10UDU": {
        "center": (48.4264, -124.0065),
        "utm_zone": "10N",
        "location_name": "Victoria / Juan de Fuca Strait",
    },
}

# Seasonal date ranges
SEASONS = {
    "winter": ("12-01", "02-28"),
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "fall": ("09-01", "11-30"),
}

# Years to download
YEARS = [2021, 2022, 2023]

# Quality thresholds based on Schroeder et al. (2020)
CLOUD_THRESHOLD = 20.0  # Maximum cloud cover percentage
VALID_PIXEL_THRESHOLD = 0.70  # Minimum valid pixel ratio after masking
MIN_SUN_ELEVATION = 30.0  # Minimum sun elevation angle (degrees)

# MGRS tile size is approximately 110km x 110km
# We use a smaller AOI around the center point for coastal areas
AOI_BUFFER_KM = 30  # Buffer around center point in km


@dataclass
class ImageMetadata:
    """Metadata for a downloaded image."""
    site: str
    season: str
    year: int
    image_id: str
    acquisition_date: str
    cloud_cover_pct: float
    valid_pixel_pct: float
    quality_score: float
    sun_elevation: float
    sun_azimuth: float
    has_label: bool
    files: Dict[str, str]


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        print("Please run 'earthengine authenticate' first.")
        raise


def create_aoi(center: Tuple[float, float], buffer_km: float = AOI_BUFFER_KM) -> ee.Geometry:
    """
    Create an Area of Interest (AOI) around a center point.

    Args:
        center: (latitude, longitude) tuple
        buffer_km: Buffer distance in kilometers

    Returns:
        ee.Geometry.Rectangle for the AOI
    """
    lat, lon = center
    # Approximate degrees per km at this latitude
    deg_per_km_lat = 1 / 111.0
    deg_per_km_lon = 1 / (111.0 * abs(cos(lat * 3.14159 / 180)))

    # Create bounding box
    min_lon = lon - (buffer_km * deg_per_km_lon)
    max_lon = lon + (buffer_km * deg_per_km_lon)
    min_lat = lat - (buffer_km * deg_per_km_lat)
    max_lat = lat + (buffer_km * deg_per_km_lat)

    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def cos(x: float) -> float:
    """Cosine function using math module."""
    import math
    return math.cos(x)


def get_sentinel2_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_threshold: float = CLOUD_THRESHOLD
) -> ee.ImageCollection:
    """
    Get Sentinel-2 L2A collection with quality filters.

    Args:
        aoi: Area of interest geometry
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cloud_threshold: Maximum cloud cover percentage

    Returns:
        Filtered ImageCollection
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    return collection


def apply_cloud_mask(image: ee.Image) -> ee.Image:
    """
    Apply SCL-based cloud masking for Sentinel-2 L2A.

    The Scene Classification Layer (SCL) provides pixel-level classification:
    - 3: Cloud shadows
    - 8: Cloud medium probability
    - 9: Cloud high probability
    - 10: Thin cirrus
    - 11: Snow/ice

    Args:
        image: Sentinel-2 L2A image

    Returns:
        Cloud-masked image
    """
    scl = image.select("SCL")

    # Create mask: True for valid pixels
    mask = (
        scl.neq(3)    # Cloud shadows
        .And(scl.neq(8))   # Cloud medium probability
        .And(scl.neq(9))   # Cloud high probability
        .And(scl.neq(10))  # Thin cirrus
        .And(scl.neq(11))  # Snow/ice
    )

    return image.updateMask(mask)


def calculate_quality_score(
    image: ee.Image,
    aoi: ee.Geometry
) -> Dict[str, float]:
    """
    Calculate image quality score based on Schroeder et al. (2020) criteria.

    Factors:
    - Cloud cover (from metadata)
    - Valid pixel percentage (after masking)
    - Sun elevation angle (higher is better for kelp detection)

    Args:
        image: Sentinel-2 image
        aoi: Area of interest

    Returns:
        Dictionary with quality metrics
    """
    # Get metadata
    cloud_cover = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    sun_zenith = image.get("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
    sun_azimuth = image.get("MEAN_SOLAR_AZIMUTH_ANGLE").getInfo()
    sun_elevation = 90 - sun_zenith  # Convert zenith to elevation

    # Calculate valid pixel percentage
    masked = apply_cloud_mask(image)

    # Count valid pixels in B4 (Red band at 10m)
    valid_count = masked.select("B4").reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi,
        scale=10,
        maxPixels=1e9
    ).get("B4").getInfo()

    total_count = image.select("B4").reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi,
        scale=10,
        maxPixels=1e9
    ).get("B4").getInfo()

    valid_pct = (valid_count / total_count) * 100 if total_count and total_count > 0 else 0

    # Calculate composite quality score
    # Weights: cloud cover (40%), valid pixels (40%), sun elevation (20%)
    cloud_score = max(0, 100 - cloud_cover)
    sun_score = max(0, min(100, (sun_elevation / 60) * 100))  # Normalize to 0-100

    quality_score = (
        0.4 * cloud_score +
        0.4 * valid_pct +
        0.2 * sun_score
    )

    return {
        "cloud_cover_pct": cloud_cover,
        "valid_pixel_pct": valid_pct,
        "sun_elevation": sun_elevation,
        "sun_azimuth": sun_azimuth,
        "quality_score": quality_score,
    }


def get_season_dates(season: str, year: int) -> Tuple[str, str]:
    """
    Get start and end dates for a season in a given year.

    Args:
        season: Season name (winter, spring, summer, fall)
        year: Year

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    start_month_day, end_month_day = SEASONS[season]

    if season == "winter":
        # Winter spans December of previous year to February
        start_date = f"{year - 1}-{start_month_day}"
        end_date = f"{year}-{end_month_day}"
    else:
        start_date = f"{year}-{start_month_day}"
        end_date = f"{year}-{end_month_day}"

    return start_date, end_date


def export_image_to_drive(
    image: ee.Image,
    aoi: ee.Geometry,
    description: str,
    folder: str,
    bands_10m: List[str],
    bands_20m: List[str],
    crs: str,
) -> Dict[str, ee.batch.Task]:
    """
    Export Sentinel-2 bands to Google Drive.

    Exports 10m and 20m bands separately, matching Mohsen's format.

    Args:
        image: Sentinel-2 image
        aoi: Area of interest
        description: Export description/filename
        folder: Google Drive folder
        bands_10m: List of 10m band names
        bands_20m: List of 20m band names
        crs: Coordinate reference system (e.g., "EPSG:32609")

    Returns:
        Dictionary of export tasks
    """
    tasks = {}

    # Export 10m bands
    task_10m = ee.batch.Export.image.toDrive(
        image=image.select(bands_10m),
        description=f"{description}_B2B3B4B8",
        folder=folder,
        fileNamePrefix=f"{description}_B2B3B4B8",
        region=aoi,
        scale=10,
        crs=crs,
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    tasks["10m"] = task_10m

    # Export 20m bands (at 10m resolution for stacking)
    task_20m = ee.batch.Export.image.toDrive(
        image=image.select(bands_20m),
        description=f"{description}_B5B6B7B8A_B11B12",
        folder=folder,
        fileNamePrefix=f"{description}_B5B6B7B8A_B11B12",
        region=aoi,
        scale=10,  # Resample to 10m
        crs=crs,
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    tasks["20m"] = task_20m

    return tasks


def download_site_season(
    site_id: str,
    season: str,
    year: int,
    output_dir: str,
    drive_folder: str = "kelp_multitemporal",
    max_images: int = 2,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Download best quality images for a site-season-year combination.

    Args:
        site_id: MGRS tile code (e.g., "T09UXQ")
        season: Season name
        year: Year
        output_dir: Local output directory (for manifest)
        drive_folder: Google Drive folder for exports
        max_images: Maximum images to download per site-season-year
        dry_run: If True, only query and score images without downloading

    Returns:
        List of image metadata dictionaries
    """
    site_config = SITES[site_id]
    center = site_config["center"]
    utm_zone = site_config["utm_zone"]

    # Create AOI
    aoi = create_aoi(center)

    # Get CRS for this UTM zone
    utm_number = utm_zone[:-1]  # Remove N/S suffix
    crs = f"EPSG:326{utm_number.zfill(2)}"

    # Get date range for season
    start_date, end_date = get_season_dates(season, year)

    print(f"  Querying {site_id} {season} {year}: {start_date} to {end_date}")

    # Query collection
    collection = get_sentinel2_collection(aoi, start_date, end_date)

    # Get list of images
    image_list = collection.toList(20)
    n_images = image_list.size().getInfo()

    if n_images == 0:
        print(f"    No images found")
        return []

    print(f"    Found {n_images} images, scoring top candidates...")

    # Score and rank images
    scored_images = []
    for i in range(min(n_images, 10)):  # Check top 10 by cloud cover
        try:
            image = ee.Image(image_list.get(i))
            image_id = image.get("system:index").getInfo()
            acquisition_time = image.get("system:time_start").getInfo()
            acquisition_date = datetime.fromtimestamp(acquisition_time / 1000).strftime("%Y-%m-%d")

            # Calculate quality metrics
            metrics = calculate_quality_score(image, aoi)

            # Check quality thresholds
            if metrics["valid_pixel_pct"] < VALID_PIXEL_THRESHOLD * 100:
                print(f"      {image_id}: Skipped (valid pixels {metrics['valid_pixel_pct']:.1f}% < {VALID_PIXEL_THRESHOLD * 100}%)")
                continue

            if metrics["sun_elevation"] < MIN_SUN_ELEVATION:
                print(f"      {image_id}: Skipped (sun elevation {metrics['sun_elevation']:.1f}° < {MIN_SUN_ELEVATION}°)")
                continue

            scored_images.append({
                "image": image,
                "image_id": image_id,
                "acquisition_date": acquisition_date,
                **metrics,
            })

            print(f"      {image_id}: score={metrics['quality_score']:.1f}, cloud={metrics['cloud_cover_pct']:.1f}%, valid={metrics['valid_pixel_pct']:.1f}%")

        except Exception as e:
            print(f"      Error scoring image {i}: {e}")
            continue

    if not scored_images:
        print(f"    No images passed quality filters")
        return []

    # Sort by quality score
    scored_images.sort(key=lambda x: x["quality_score"], reverse=True)

    # Select top N
    selected = scored_images[:max_images]
    print(f"    Selected {len(selected)} images for download")

    if dry_run:
        return [
            {
                "site": site_id,
                "season": season,
                "year": year,
                "image_id": img["image_id"],
                "acquisition_date": img["acquisition_date"],
                "cloud_cover_pct": img["cloud_cover_pct"],
                "valid_pixel_pct": img["valid_pixel_pct"],
                "quality_score": img["quality_score"],
                "sun_elevation": img["sun_elevation"],
                "has_label": False,
                "status": "dry_run",
            }
            for img in selected
        ]

    # Export selected images
    downloaded = []
    bands_10m = ["B2", "B3", "B4", "B8"]
    bands_20m = ["B5", "B6", "B7", "B8A", "B11", "B12"]

    for img_data in selected:
        image = img_data["image"]
        image_id = img_data["image_id"]

        # Create filename
        filename = f"{image_id}_{site_id}_{season}_{year}"

        # Start export tasks
        try:
            tasks = export_image_to_drive(
                image=image,
                aoi=aoi,
                description=filename,
                folder=drive_folder,
                bands_10m=bands_10m,
                bands_20m=bands_20m,
                crs=crs,
            )

            # Start tasks
            for task_name, task in tasks.items():
                task.start()
                print(f"      Started export: {filename}_{task_name}")

            downloaded.append({
                "site": site_id,
                "season": season,
                "year": year,
                "image_id": image_id,
                "acquisition_date": img_data["acquisition_date"],
                "cloud_cover_pct": img_data["cloud_cover_pct"],
                "valid_pixel_pct": img_data["valid_pixel_pct"],
                "quality_score": img_data["quality_score"],
                "sun_elevation": img_data["sun_elevation"],
                "sun_azimuth": img_data["sun_azimuth"],
                "has_label": False,
                "files": {
                    "10m_bands": f"{filename}_B2B3B4B8.tif",
                    "20m_bands": f"{filename}_B5B6B7B8A_B11B12.tif",
                },
                "status": "exporting",
            })

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"      Error exporting {image_id}: {e}")
            continue

    return downloaded


def download_all_sites(
    output_dir: str,
    sites: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    max_images_per_combo: int = 2,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Download imagery for all site-season-year combinations.

    Args:
        output_dir: Output directory for manifest
        sites: List of site IDs (default: all sites)
        seasons: List of seasons (default: all seasons)
        years: List of years (default: 2021-2023)
        max_images_per_combo: Max images per site-season-year
        dry_run: If True, only query without downloading

    Returns:
        List of all download metadata
    """
    sites = sites or list(SITES.keys())
    seasons = seasons or list(SEASONS.keys())
    years = years or YEARS

    all_downloads = []

    for site_id in sites:
        print(f"\nProcessing site: {site_id} ({SITES[site_id]['location_name']})")

        for year in years:
            for season in seasons:
                downloads = download_site_season(
                    site_id=site_id,
                    season=season,
                    year=year,
                    output_dir=output_dir,
                    max_images=max_images_per_combo,
                    dry_run=dry_run,
                )
                all_downloads.extend(downloads)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save download manifest
    manifest = {
        "download_id": f"dl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "config": {
            "sites": sites,
            "seasons": seasons,
            "years": years,
            "max_images_per_combo": max_images_per_combo,
            "cloud_threshold": CLOUD_THRESHOLD,
            "valid_pixel_threshold": VALID_PIXEL_THRESHOLD,
            "min_sun_elevation": MIN_SUN_ELEVATION,
        },
        "summary": {
            "total_images": len(all_downloads),
            "by_site": {
                site: len([d for d in all_downloads if d["site"] == site])
                for site in sites
            },
            "by_season": {
                season: len([d for d in all_downloads if d["season"] == season])
                for season in seasons
            },
            "by_year": {
                year: len([d for d in all_downloads if d["year"] == year])
                for year in years
            },
        },
        "images": all_downloads,
    }

    manifest_path = output_path / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"Total images: {len(all_downloads)}")
    print(f"Manifest saved: {manifest_path}")

    if not dry_run:
        print(f"\nExports started. Check Google Earth Engine Tasks:")
        print(f"  https://code.earthengine.google.com/tasks")
        print(f"\nOnce complete, download from Google Drive folder: kelp_multitemporal")

    return all_downloads


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download multi-temporal Sentinel-2 imagery for BC kelp sites"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/bc_sentinel2_multitemporal/raw/",
        help="Output directory for manifest and metadata",
    )
    parser.add_argument(
        "--sites",
        type=str,
        nargs="+",
        default=None,
        help="Specific sites to download (default: all)",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        choices=["winter", "spring", "summer", "fall"],
        default=None,
        help="Specific seasons to download (default: all)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Specific years to download (default: 2021-2023)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2,
        help="Maximum images per site-season-year combination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and score images without downloading",
    )

    args = parser.parse_args()

    # Initialize GEE
    initialize_gee()

    # Run download
    download_all_sites(
        output_dir=args.output,
        sites=args.sites,
        seasons=args.seasons,
        years=args.years,
        max_images_per_combo=args.max_images,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
