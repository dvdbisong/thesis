# LA Kelp Detection - Full Research Implementation Plan

**Scope:** Full PhD thesis experiments (not prototype)
**Last Updated:** March 14, 2026
**Root:** `/Users/ebisong/Documents/code/uvic/thesis`

---

## Research Questions Addressed

| RQ | Question | How Addressed |
|----|----------|---------------|
| RQ1 | Adapt without extensive labeled data | LA learns from sparse feedback |
| RQ2 | Adaptive selection across heterogeneous regions | 10 DL heuristics + LOSO evaluation |
| RQ3 | Improve with sparse, async labels | Component 3: Sparse Label Handling |
| **RQ4** | **Fuse heterogeneous data (10m-5km)** | **Multi-resolution + cross-sensor fusion** |

---

## RQ4: Data Fusion Strategy

### Two Fusion Dimensions

**Dimension 1: Multi-Resolution Auxiliary Fusion**
```
Sentinel-2 (10m) + Bathymetry (30m-1km) + Substrate (30m) + [Future: SST (1-5km), Tide]
```
- Resample auxiliary data to match imagery resolution
- Handle missing data gracefully (masking, imputation)
- Context vectors incorporate multi-resolution features

**Dimension 2: Cross-Sensor Fusion**
```
Sentinel-2 (10m, 12 bands) ↔ Landsat (30m, RGB)
```

**Cross-Sensor Challenge:**
- BC tiles: 12 bands, 512×512, float32
- Mohsen's models: Expect 12-band input (cannot directly use on RGB)
- Figshare Landsat: RGB only (3 bands), 1024×1024

**Solution: Single RGB Model for Context Transfer Testing**
1. Train single 3-band model on ALL BC Sentinel-2 sites (bands 4, 3, 2)
2. Test on BC RGB → establishes RGB-only baseline
3. Test on Figshare Landsat RGB → tests if LA's context understanding transfers
4. Goal: Demonstrate that LA's memory bank similarity matching generalizes across sensors

**Why Single Model (Not LOO):**
- The research goal is NOT comparing 12-band vs RGB performance
- The goal IS demonstrating LA's adaptive selection based on context
- For cross-sensor: We test if context representations transfer, not heuristic selection
- Primary heuristic selection is demonstrated on BC with 10 site-specific 12-band models
- Simpler experimental design with cleaner interpretation

### Scientific Approach

1. **Early Fusion:** Concatenate resampled auxiliary bands as input channels
2. **Late Fusion:** Separate processing, fuse at decision/context level
3. **Adaptive Fusion:** LA learns which fusion strategy works best per context

### Current vs Future Data

| Data Source | Resolution | Status | Used For |
|-------------|------------|--------|----------|
| Sentinel-2 (BC) | 10m | Available | Primary training/testing |
| Bathymetry | ~30m | Available (Mohsen) | Auxiliary feature |
| Substrate | ~30m | Available (Mohsen) | Auxiliary feature |
| Landsat (Figshare) | 30m | Available | Cross-sensor testing |
| SST | 1-5km | Future | Context features |
| Tide Height | Point | Future | Context features |

---

## Full Experiment Design

### Scale

| Dataset | Tiles/Images | Sites | Labels | Purpose |
|---------|--------------|-------|--------|---------|
| BC Sentinel-2 (Original) | ~7 images, ~300 tiles | 7 BC sites | Yes | Primary training/validation |
| BC Sentinel-2 (Multi-temporal) | ~80-100 images, ~3000+ tiles | 7 BC sites | No | Layer 1 testing, sparse label simulation |
| Figshare Landsat | 432 images | California | Yes | Cross-sensor testing |
| Falkland Islands | TBD | Future | TBD | Cross-hemisphere |

**Note:** Multi-temporal data enables empirical testing of dual-layer MSE/PSE framework with truly sparse labels (~7% labeled).

### Experimental Matrix

```
6 LA variants × 10 DL heuristics × 4 components × staged ablations
= Comprehensive evaluation via staged approach
```

---

## Phase 0: EDA & Data Preparation

### 0.1 BC Data Analysis
- [ ] Load and visualize 12-band Sentinel-2 tiles (512×512, float32)
- [ ] **Confirmed:** Tile-level masks already paired with images (Mohsen preprocessed)
- [ ] Verify image-mask alignment across all 10 sites
- [ ] Analyze band distributions, kelp prevalence per site
- [ ] Validate Bathymetry and Substrate auxiliary layers

### 0.2 Figshare Data Analysis
- [ ] Convert VIA JSON polygons to binary masks
- [ ] Understand Landsat RGB composite limitations
- [ ] Compare spectral characteristics with Sentinel-2

### 0.3 Data Fusion Preprocessing
- [ ] Resample auxiliary data (Bathymetry, Substrate) to tile resolution
- [ ] Document missing data patterns
- [ ] Create unified data loader supporting both datasets

### 0.4 High-Quality 4K Sample Maps
- [ ] Generate publication-ready 4K sample maps (400 DPI, ~3840×2160)
- [ ] **BC Sentinel-2:** RGB composite (bands 4, 3, 2) with kelp mask overlay for all 10 sites
- [ ] **Figshare Landsat:** True color RGB with kelp polygon boundaries
- [ ] Include cartographic elements: scale bar, north arrow, site label, coordinate grid
- [ ] Save as PNG (raster) and PDF (vector) for publication flexibility
- [ ] Store in `results/raw/phase_0_eda/sample_maps/`

### Deliverables
- `src/notebooks/data_eda.ipynb`
- `data/bc_sentinel2/` - BC tiles with masks (already aligned)
- `data/figshare_landsat/` - Cross-sensor dataset
- `results/raw/phase_0_eda/sample_maps/` - 4K publication-ready maps

---

## Phase 0.5: Multi-Temporal Data Acquisition & Preprocessing

### Rationale

**Problem:** Current BC data has single observation per site (all summer). This prevents empirical testing of Layer 1 (Environment Dynamics) in the dual-layer MSE/PSE framework.

**Solution:** Download additional Sentinel-2 imagery for the same 7 sites across multiple seasons. These images will be UNLABELED, but the LA framework is designed for sparse labels (RQ3).

**Scientific Value:**
- Enables empirical Layer 1 testing (seasonal/climate heuristic adaptation)
- Validates RQ3 (learning from truly sparse feedback)
- Creates realistic operational monitoring scenario
- Enables phenological analysis of kelp extent

### 0.5.1 Target Sites and Temporal Coverage

**Sites (from existing labeled data):**

| Site Code | UTM Zone | Labeled Date | Location Context |
|-----------|----------|--------------|------------------|
| T09UXQ | 9U | July 27, 2021 | BC Coast |
| T09UYQ | 9U | August 6, 2022 | BC Coast |
| T10UCU | 10U | August 6, 2022 | BC Coast |
| T09UXS | 9U | August 4, 2023 | BC Coast |
| T10UCA | 10U | August 16, 2023 | BC Coast |
| T09UWT | 9U | August 19, 2023 | BC Coast |
| T09UUU | 9U | September 2, 2023 | BC Coast |

**Target Temporal Coverage:**

| Season | Months | Target Images/Site | Rationale |
|--------|--------|-------------------|-----------|
| Winter | Dec-Feb | 1-2 | Low kelp canopy, high turbidity, storm disturbance |
| Spring | Mar-May | 2-3 | Rapid growth phase, increasing light |
| Summer | Jun-Aug | 2-3 (includes labeled) | Peak canopy, optimal conditions |
| Fall | Sep-Nov | 2-3 | Senescence onset, canopy thinning |

**Years:** 2021, 2022, 2023 (match existing data range)

**Estimated Volume:**
```
7 sites × 4 seasons × ~2 images/season × 3 years = ~168 candidate images
After quality filtering (~50% usable): ~80-100 images
+ 7 original labeled images: ~90-110 total images
```

### 0.5.2 Download Strategy

#### Platform: Google Earth Engine (GEE)

**Why GEE:**
- Free access to Sentinel-2 archive
- Scriptable bulk download
- Built-in cloud masking algorithms
- Consistent preprocessing

#### GEE Download Script

```python
# src/preprocessing/gee_download.py

import ee
import geemap
from datetime import datetime
from typing import List, Dict

# Initialize Earth Engine
ee.Initialize()

# Site definitions (bounding boxes from existing tiles)
SITES = {
    "T09UXQ": {"bounds": [...], "utm_zone": "9N"},
    "T09UYQ": {"bounds": [...], "utm_zone": "9N"},
    "T10UCU": {"bounds": [...], "utm_zone": "10N"},
    "T09UXS": {"bounds": [...], "utm_zone": "9N"},
    "T10UCA": {"bounds": [...], "utm_zone": "10N"},
    "T09UWT": {"bounds": [...], "utm_zone": "9N"},
    "T09UUU": {"bounds": [...], "utm_zone": "9N"},
}

# Seasonal date ranges
SEASONS = {
    "winter": [("12-01", "02-28")],
    "spring": [("03-01", "05-31")],
    "summer": [("06-01", "08-31")],
    "fall": [("09-01", "11-30")],
}

YEARS = [2021, 2022, 2023]

def get_sentinel2_collection(site_bounds: ee.Geometry,
                              start_date: str,
                              end_date: str,
                              cloud_threshold: float = 20.0) -> ee.ImageCollection:
    """
    Get Sentinel-2 L2A collection with quality filters.

    Args:
        site_bounds: GEE geometry for the site
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cloud_threshold: Maximum cloud cover percentage

    Returns:
        Filtered ImageCollection
    """
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(site_bounds)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .sort("CLOUDY_PIXEL_PERCENTAGE"))

    return collection


def apply_cloud_mask(image: ee.Image) -> ee.Image:
    """Apply SCL-based cloud masking for Sentinel-2 L2A."""
    scl = image.select("SCL")

    # SCL classes to mask: clouds, cloud shadows, cirrus
    mask = (scl.neq(3)   # Cloud shadows
            .And(scl.neq(8))   # Cloud medium probability
            .And(scl.neq(9))   # Cloud high probability
            .And(scl.neq(10))  # Thin cirrus
            .And(scl.neq(11))) # Snow/ice

    return image.updateMask(mask)


def calculate_quality_score(image: ee.Image, site_bounds: ee.Geometry) -> float:
    """
    Calculate image quality score based on Schroeder et al. (2020) criteria.

    Factors:
    - Cloud cover (from metadata)
    - Valid pixel percentage (after masking)
    - Sun elevation angle
    - View angle

    Returns:
        Quality score 0-100
    """
    # Cloud cover from metadata
    cloud_cover = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    # Valid pixel percentage
    masked = apply_cloud_mask(image)
    valid_pixels = masked.select("B4").reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=site_bounds,
        scale=10
    ).get("B4").getInfo()

    total_pixels = image.select("B4").reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=site_bounds,
        scale=10
    ).get("B4").getInfo()

    valid_pct = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Sun elevation (higher is better for kelp detection)
    sun_elevation = image.get("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
    sun_score = max(0, 100 - sun_elevation)  # Lower zenith = higher sun = better

    # Combined score (weighted)
    quality_score = (
        0.4 * (100 - cloud_cover) +
        0.4 * valid_pct +
        0.2 * sun_score
    )

    return quality_score


def download_site_season(site_id: str,
                          season: str,
                          year: int,
                          output_dir: str,
                          max_images: int = 3) -> List[Dict]:
    """
    Download best quality images for a site-season-year combination.

    Returns:
        List of downloaded image metadata
    """
    site_config = SITES[site_id]
    site_bounds = ee.Geometry.Rectangle(site_config["bounds"])

    # Get date range for season
    start_month, end_month = SEASONS[season][0]
    start_date = f"{year}-{start_month}"
    end_date = f"{year}-{end_month}"

    # Query collection
    collection = get_sentinel2_collection(site_bounds, start_date, end_date)

    # Get list of images
    image_list = collection.toList(20)
    n_images = image_list.size().getInfo()

    if n_images == 0:
        print(f"No images found for {site_id} {season} {year}")
        return []

    # Score and rank images
    scored_images = []
    for i in range(min(n_images, 10)):  # Check top 10 by cloud cover
        image = ee.Image(image_list.get(i))
        score = calculate_quality_score(image, site_bounds)
        image_id = image.get("system:index").getInfo()
        scored_images.append({"image": image, "score": score, "id": image_id})

    # Sort by quality score
    scored_images.sort(key=lambda x: x["score"], reverse=True)

    # Download top N
    downloaded = []
    for img_data in scored_images[:max_images]:
        image = img_data["image"]
        image_id = img_data["id"]

        # Select bands matching Mohsen's format
        bands_10m = ["B2", "B3", "B4", "B8"]
        bands_20m = ["B5", "B6", "B7", "B8A", "B11", "B12"]

        # Export 10m bands
        export_image_10m = image.select(bands_10m)
        export_image_20m = image.select(bands_20m)

        # Define export parameters
        export_params = {
            "region": site_bounds,
            "scale": 10,
            "crs": f"EPSG:326{site_config['utm_zone'][:-1]}",  # UTM zone
            "fileFormat": "GeoTIFF",
        }

        # Export to Google Drive (then download locally)
        filename = f"{image_id}_{site_id}_{season}_{year}"
        # ... export logic ...

        downloaded.append({
            "site": site_id,
            "season": season,
            "year": year,
            "image_id": image_id,
            "quality_score": img_data["score"],
            "filename": filename,
        })

    return downloaded


def download_all_sites():
    """Main function to download all site-season-year combinations."""
    all_downloads = []

    for site_id in SITES.keys():
        for year in YEARS:
            for season in SEASONS.keys():
                print(f"Processing {site_id} {season} {year}...")
                downloads = download_site_season(
                    site_id=site_id,
                    season=season,
                    year=year,
                    output_dir="data/bc_sentinel2_multitemporal/raw/",
                    max_images=2
                )
                all_downloads.extend(downloads)

    # Save download manifest
    import json
    with open("data/bc_sentinel2_multitemporal/download_manifest.json", "w") as f:
        json.dump(all_downloads, f, indent=2)

    return all_downloads
```

#### Quality Criteria (Based on Schroeder et al., 2020)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Cloud cover | < 20% | Metadata filter |
| Valid pixels | > 70% | After SCL masking |
| Sun elevation | > 30° | Avoid low-angle artifacts |
| Glint risk | Low | Based on sun-view geometry |

#### Download Manifest Schema

```json
{
  "download_id": "dl_20260315_001",
  "timestamp": "2026-03-15T10:00:00Z",
  "images": [
    {
      "site": "T09UXQ",
      "season": "winter",
      "year": 2021,
      "image_id": "20211215T192911_20211215T193000_T09UXQ",
      "acquisition_date": "2021-12-15",
      "cloud_cover_pct": 12.3,
      "valid_pixel_pct": 85.2,
      "quality_score": 78.5,
      "sun_elevation": 22.1,
      "has_label": false,
      "files": {
        "10m_bands": "..._B2B3B4B8.tif",
        "20m_bands": "..._B5B6B7B8A_B11B12.tif"
      }
    }
  ]
}
```

### 0.5.3 Preprocessing Pipeline Implementation

Since we don't have access to Mohsen's preprocessing pipeline, we implement our own to match his output format.

#### Target Output Format (Matching Existing Data)

```
data/bc_sentinel2_multitemporal/
├── Tiles/
│   └── {image_id}_{site}/
│       ├── images/
│       │   └── tile_{N}_image.tiff  # 12 bands, 512×512
│       └── masks/
│           └── tile_{N}_mask.tiff   # Empty (no labels)
├── Masks/
│   └── {image_id}_{site}/
│       ├── {image_id}_{site}_B2B3B4B8.tif
│       ├── {image_id}_{site}_B5B6B7B8A_B11B12.tif
│       ├── {image_id}_{site}_Bathymetry.tif
│       ├── {image_id}_{site}_Substrate.tif
│       └── {image_id}_{site}_Mask.tif  # All zeros (no labels)
└── download_manifest.json
```

#### Preprocessing Steps

```python
# src/preprocessing/preprocess_multitemporal.py

import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from pathlib import Path
from typing import Tuple, List

class Sentinel2Preprocessor:
    """
    Preprocessor to match Mohsen's data format.

    Input: Raw Sentinel-2 bands from GEE
    Output: 12-band tiles (512×512) with auxiliary layers
    """

    TILE_SIZE = 512
    TARGET_BANDS = [
        "B2", "B3", "B4", "B8",           # 10m bands (indices 0-3)
        "B5", "B6", "B7", "B8A", "B11", "B12",  # 20m bands resampled (indices 4-9)
        "Substrate", "Bathymetry"          # Auxiliary (indices 10-11)
    ]

    def __init__(self, auxiliary_dir: str):
        """
        Args:
            auxiliary_dir: Path to existing auxiliary data (Bathymetry, Substrate)
        """
        self.auxiliary_dir = Path(auxiliary_dir)

    def resample_20m_to_10m(self, src_path: str, dst_path: str):
        """Resample 20m bands to 10m resolution using bilinear interpolation."""
        with rasterio.open(src_path) as src:
            # Calculate new dimensions
            new_height = src.height * 2
            new_width = src.width * 2

            # Update transform
            new_transform = src.transform * src.transform.scale(0.5, 0.5)

            # Read and resample each band
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.bilinear
            )

            # Write output
            profile = src.profile.copy()
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
            })

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)

    def align_auxiliary_to_image(self,
                                  aux_path: str,
                                  reference_path: str,
                                  output_path: str):
        """
        Align auxiliary data (Bathymetry/Substrate) to Sentinel-2 image grid.

        Steps:
        1. Reproject to same CRS
        2. Resample to 10m
        3. Clip to image extent
        """
        with rasterio.open(reference_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_bounds = ref.bounds
            ref_shape = (ref.height, ref.width)

        with rasterio.open(aux_path) as aux:
            # Allocate output array
            aux_aligned = np.zeros(ref_shape, dtype=aux.dtypes[0])

            # Reproject
            reproject(
                source=rasterio.band(aux, 1),
                destination=aux_aligned,
                src_transform=aux.transform,
                src_crs=aux.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )

        # Write aligned auxiliary
        profile = {
            "driver": "GTiff",
            "height": ref_shape[0],
            "width": ref_shape[1],
            "count": 1,
            "dtype": aux_aligned.dtype,
            "crs": ref_crs,
            "transform": ref_transform,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(aux_aligned, 1)

    def stack_bands(self,
                    bands_10m_path: str,
                    bands_20m_resampled_path: str,
                    substrate_path: str,
                    bathymetry_path: str,
                    output_path: str):
        """
        Stack all 12 bands into single GeoTIFF.

        Band order:
        1-4: B2, B3, B4, B8 (10m)
        5-10: B5, B6, B7, B8A, B11, B12 (resampled to 10m)
        11: Substrate
        12: Bathymetry
        """
        with rasterio.open(bands_10m_path) as src_10m:
            profile = src_10m.profile.copy()
            data_10m = src_10m.read()  # Shape: (4, H, W)

        with rasterio.open(bands_20m_resampled_path) as src_20m:
            data_20m = src_20m.read()  # Shape: (6, H, W)

        with rasterio.open(substrate_path) as src_sub:
            data_substrate = src_sub.read(1)  # Shape: (H, W)

        with rasterio.open(bathymetry_path) as src_bath:
            data_bathymetry = src_bath.read(1)  # Shape: (H, W)

        # Stack all bands
        stacked = np.concatenate([
            data_10m,                          # Bands 1-4
            data_20m,                          # Bands 5-10
            data_substrate[np.newaxis, :, :],  # Band 11
            data_bathymetry[np.newaxis, :, :], # Band 12
        ], axis=0)

        # Update profile
        profile.update({"count": 12})

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stacked)

    def create_tiles(self,
                     stacked_image_path: str,
                     output_dir: str,
                     overlap: int = 0) -> List[str]:
        """
        Tile the full image into 512×512 chips.

        Args:
            stacked_image_path: Path to 12-band stacked image
            output_dir: Directory for output tiles
            overlap: Pixel overlap between tiles (default 0)

        Returns:
            List of tile paths
        """
        output_dir = Path(output_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)

        tile_paths = []

        with rasterio.open(stacked_image_path) as src:
            height, width = src.height, src.width
            stride = self.TILE_SIZE - overlap

            tile_idx = 0
            for row in range(0, height - self.TILE_SIZE + 1, stride):
                for col in range(0, width - self.TILE_SIZE + 1, stride):
                    # Read tile
                    window = Window(col, row, self.TILE_SIZE, self.TILE_SIZE)
                    tile_data = src.read(window=window)

                    # Skip tiles with too many nodata pixels
                    valid_ratio = np.mean(tile_data[0] != 0)
                    if valid_ratio < 0.5:
                        continue

                    # Save tile
                    tile_path = output_dir / "images" / f"tile_{tile_idx}_image.tiff"
                    tile_profile = src.profile.copy()
                    tile_profile.update({
                        "height": self.TILE_SIZE,
                        "width": self.TILE_SIZE,
                        "transform": rasterio.windows.transform(window, src.transform),
                    })

                    with rasterio.open(tile_path, "w", **tile_profile) as dst:
                        dst.write(tile_data)

                    # Create empty mask (no labels)
                    mask_path = output_dir / "masks" / f"tile_{tile_idx}_mask.tiff"
                    mask_profile = tile_profile.copy()
                    mask_profile.update({"count": 1, "dtype": "uint8"})

                    with rasterio.open(mask_path, "w", **mask_profile) as dst:
                        dst.write(np.zeros((1, self.TILE_SIZE, self.TILE_SIZE), dtype=np.uint8))

                    tile_paths.append(str(tile_path))
                    tile_idx += 1

        return tile_paths

    def process_image(self, image_config: dict, output_base: str):
        """
        Full preprocessing pipeline for a single image.

        Args:
            image_config: Dict with paths and metadata
            output_base: Base output directory
        """
        image_id = image_config["image_id"]
        site = image_config["site"]

        output_dir = Path(output_base) / f"{image_id}_{site}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Resample 20m bands to 10m
        bands_20m_resampled = output_dir / "bands_20m_resampled.tif"
        self.resample_20m_to_10m(
            image_config["files"]["20m_bands"],
            str(bands_20m_resampled)
        )

        # Step 2: Align auxiliary data
        substrate_aligned = output_dir / "substrate_aligned.tif"
        bathymetry_aligned = output_dir / "bathymetry_aligned.tif"

        # Use existing auxiliary from labeled data (same site)
        existing_aux_dir = self.auxiliary_dir / f"*{site}*"
        # ... find and align auxiliary data ...

        # Step 3: Stack all bands
        stacked_path = output_dir / "stacked_12band.tif"
        self.stack_bands(
            image_config["files"]["10m_bands"],
            str(bands_20m_resampled),
            str(substrate_aligned),
            str(bathymetry_aligned),
            str(stacked_path)
        )

        # Step 4: Create tiles
        tiles_dir = Path(output_base).parent / "Tiles" / f"{image_id}_{site}"
        tile_paths = self.create_tiles(str(stacked_path), str(tiles_dir))

        return {
            "image_id": image_id,
            "site": site,
            "n_tiles": len(tile_paths),
            "tiles_dir": str(tiles_dir),
        }
```

### 0.5.4 Auxiliary Data Handling

**Challenge:** New images need Bathymetry and Substrate layers, but these are static (don't change over time).

**Solution:** Reuse auxiliary data from existing labeled images for the same site.

```python
# Map site codes to existing auxiliary data paths
AUXILIARY_MAPPING = {
    "T09UXQ": "data/bc_sentinel2/Masks/20210727T191911_20210727T192721_T09UXQ/",
    "T09UYQ": "data/bc_sentinel2/Masks/20220806T191919_20220806T192707_T09UYQ/",
    "T10UCU": "data/bc_sentinel2/Masks/20220806T191919_20220806T192707_T10UCU/",
    "T09UXS": "data/bc_sentinel2/Masks/20230804T192909_20230804T192942_T09UXS/",
    "T10UCA": "data/bc_sentinel2/Masks/20230816T191911_20230816T192348_T10UCA/",
    "T09UWT": "data/bc_sentinel2/Masks/20230819T192911_20230819T193100_T09UWT/",
    "T09UUU": "data/bc_sentinel2/Masks/20230902T195919_20230902T195917_T09UUU/",
}
```

### 0.5.5 Data Organization

```
data/
├── bc_sentinel2/                    # Original labeled data (Mohsen)
│   ├── Tiles/
│   └── Masks/
│
├── bc_sentinel2_multitemporal/      # NEW: Multi-temporal unlabeled
│   ├── raw/                         # Raw downloads from GEE
│   │   └── {image_id}/
│   │       ├── B2B3B4B8.tif
│   │       └── B5B6B7B8A_B11B12.tif
│   │
│   ├── processed/                   # Preprocessed (aligned, stacked)
│   │   └── {image_id}_{site}/
│   │       └── stacked_12band.tif
│   │
│   ├── Tiles/                       # Tiled (matches Mohsen format)
│   │   └── {image_id}_{site}/
│   │       ├── images/
│   │       │   └── tile_{N}_image.tiff
│   │       └── masks/
│   │           └── tile_{N}_mask.tiff  # All zeros (no labels)
│   │
│   ├── download_manifest.json       # Metadata for all downloads
│   └── preprocessing_log.json       # Processing status
│
└── figshare_landsat/                # Cross-sensor data
```

### 0.5.6 Evaluation Strategy for Unlabeled Data

Since multi-temporal images have no ground truth labels, use proxy metrics:

| Metric | What It Measures | Implementation |
|--------|------------------|----------------|
| **LA Convergence** | Policy stability | Entropy of probability distribution over time |
| **Temporal Coherence** | Prediction consistency | Kelp extent variance within short time windows |
| **Heuristic Agreement** | Cross-model consistency | IoU between heuristics on same tile |
| **Phenological Plausibility** | Matches known patterns | Summer peak, winter minimum (literature) |
| **Held-out Validation** | Sparse label performance | Reserve some labeled images for testing |

```python
# src/evaluation/unlabeled_metrics.py

def temporal_coherence(predictions: List[np.ndarray],
                       timestamps: List[datetime],
                       window_days: int = 30) -> float:
    """
    Measure prediction consistency within time windows.

    High-quality LA should produce similar predictions for
    images close in time (kelp doesn't change rapidly).
    """
    coherence_scores = []

    for i, (pred_i, time_i) in enumerate(zip(predictions, timestamps)):
        for j, (pred_j, time_j) in enumerate(zip(predictions, timestamps)):
            if i >= j:
                continue
            delta_days = abs((time_j - time_i).days)
            if delta_days <= window_days:
                # Calculate IoU between predictions
                iou = compute_iou(pred_i, pred_j)
                coherence_scores.append(iou)

    return np.mean(coherence_scores) if coherence_scores else 0.0


def phenological_plausibility(monthly_extents: Dict[int, float],
                               expected_pattern: str = "temperate") -> float:
    """
    Check if predicted kelp extent matches expected seasonal pattern.

    Temperate kelp forests: peak in summer/fall, minimum in winter.
    """
    if expected_pattern == "temperate":
        # Expected: summer > winter
        summer_extent = np.mean([monthly_extents.get(m, 0) for m in [6, 7, 8]])
        winter_extent = np.mean([monthly_extents.get(m, 0) for m in [12, 1, 2]])

        if summer_extent > winter_extent:
            return 1.0  # Matches expected pattern
        else:
            return winter_extent / summer_extent if summer_extent > 0 else 0.0

    return 0.5  # Unknown pattern
```

### 0.5.7 Integration with Dual-Layer Framework

The multi-temporal data enables empirical testing of Layer 1 (Environment Dynamics):

```
Experimental Design:
────────────────────────────────────────────────────────────────────────
1. Train LA on labeled summer images (original 7)
2. Deploy LA on multi-temporal unlabeled images
3. Observe: Does LA change heuristic selection based on season?
4. Measure: Do seasonal selections match expected patterns?
   - Winter: Turbidity-robust heuristics selected more often?
   - Summer: Optimal-condition heuristics selected?
5. Evaluate: Proxy metrics (coherence, plausibility, agreement)
────────────────────────────────────────────────────────────────────────
```

### 0.5.8 Deliverables

- [ ] `src/preprocessing/gee_download.py` - GEE download script
- [ ] `src/preprocessing/preprocess_multitemporal.py` - Preprocessing pipeline
- [ ] `src/preprocessing/tile_creator.py` - Tiling utilities
- [ ] `src/evaluation/unlabeled_metrics.py` - Proxy evaluation metrics
- [ ] `data/bc_sentinel2_multitemporal/` - Multi-temporal dataset
- [ ] `data/bc_sentinel2_multitemporal/download_manifest.json` - Metadata

### 0.5.9 Makefile Additions

```makefile
# ========================= MULTI-TEMPORAL DATA =========================
download-multitemporal:
	$(PYTHON) src/preprocessing/gee_download.py --output data/bc_sentinel2_multitemporal/raw/

preprocess-multitemporal:
	$(PYTHON) src/preprocessing/preprocess_multitemporal.py \
		--input data/bc_sentinel2_multitemporal/raw/ \
		--output data/bc_sentinel2_multitemporal/Tiles/ \
		--auxiliary data/bc_sentinel2/Masks/

validate-multitemporal:
	$(PYTHON) src/preprocessing/validate_tiles.py --dir data/bc_sentinel2_multitemporal/Tiles/

multitemporal-all: download-multitemporal preprocess-multitemporal validate-multitemporal
```

---

## Phase 1: Model Integration & Baseline

### 1.1 Heuristic Pool Wrapper

**What is a wrapper?**
A software abstraction layer that provides a unified interface to all heuristics (models). This allows the LA framework to call any heuristic uniformly without knowing implementation details.

```python
class HeuristicPool:
    """Unified interface for all detection heuristics"""
    def __init__(self, model_dir: str):
        self.models = {}
        for site in SITES:
            self.models[site] = load_model(f"{model_dir}/{site}.pt")

    def predict(self, tile: np.ndarray, heuristic_id: str) -> np.ndarray:
        """LA framework calls this - same interface for all heuristics"""
        model = self.models[heuristic_id]
        preprocessed = self._preprocess(tile)
        return model(preprocessed)

    def list_heuristics(self) -> List[str]:
        return list(self.models.keys())
```

- [ ] Implement `HeuristicPool` class with unified `predict()` interface
- [ ] Load 10 site-specific UNet-MaxViT models (12-band)
- [ ] Support variable input channels (with/without auxiliary)
- [ ] Add RGB model to pool for cross-sensor experiments

### 1.2 Cross-Sensor RGB Model Training

**Training Protocol: Single Model on All BC Sites**

| Aspect | Specification |
|--------|---------------|
| **Architecture** | UNet-MaxViT, `in_channels=3` |
| **Input** | BC Sentinel-2 bands 4, 3, 2 (RGB composite) |
| **Training Split** | 80/10/10 train/val/test across all BC tiles |
| **Loss** | Dice Loss + Binary Cross-Entropy (combined) |
| **Optimizer** | AdamW, lr=1e-4, weight_decay=1e-5 |
| **Scheduler** | CosineAnnealingLR or ReduceLROnPlateau |
| **Epochs** | 100 with early stopping (patience=10) |
| **Augmentation** | HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter |
| **Batch Size** | 8-16 (GPU memory dependent) |
| **Validation Metric** | IoU on validation set |

- [ ] Implement `src/training/train_rgb_model.py`
- [ ] Train single RGB model on all BC sites
- [ ] Validate on BC RGB subset
- [ ] Save to `models/rgb/rgb_all_sites.pt`

### 1.3 RunPod Deployment

**Deployment Strategy: Hybrid Batch/Sequential**

Operational kelp monitoring workflow:
1. **Batch Download:** Fetch all new Sentinel-2 tiles for BC coast (arrives every 5 days)
2. **Sequential LA Processing:** For each tile:
   - Extract context features
   - LA selects heuristic based on `p_final(c)`
   - Run selected model inference
   - Store prediction in buffer (awaiting validation)
3. **Delayed Update:** When validation labels arrive, update LA probabilities (can be batched)

**Deployment Scripts:**
- [ ] `scripts/runpod/deploy.sh` - Environment setup, model deployment
- [ ] `scripts/runpod/inference_server.py` - FastAPI endpoint for inference
- [ ] `scripts/runpod/la_inference_pipeline.py` - Full LA processing pipeline
- [ ] `scripts/runpod/batch_download.py` - Fetch imagery batches
- [ ] `scripts/runpod/delayed_update.py` - Process validation label batches
- [ ] Docker configuration for reproducibility

**Note:** GPU batching within a heuristic (process multiple tiles with same model), but LA decision-making is sequential per tile.

### 1.4 Baseline Experiments
- [ ] Per-model performance on held-out sites
- [ ] Random selection baseline
- [ ] Oracle (best model per tile) upper bound

---

## Phase 2: Component 1 - Global Policy Learning

### 2.1 Single-Objective (IoU)
- [ ] Run all 6 LA variants: LR-I, LR-P, VSLA, Pursuit, DiscretizedPursuit, SERI
- [ ] Metrics: convergence speed, final IoU, entropy, probability evolution

### 2.2 Multi-Objective Extension

```python
R_total = w1*R_IoU + w2*R_speed + w3*R_uncertainty + w4*R_carbon
```

#### Carbon Computation Strategy

**Measurement:** Use CodeCarbon library for accurate, publishable CO2 tracking
```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(log_level="error")
tracker.start()
prediction = model(tile)
emissions_kg = tracker.stop()  # Returns kg CO2 equivalent
```

**Reward Function:** Use relative efficiency (inference time) for stable LA updates
```python
R_carbon = 1 - (inference_time_i / max_inference_time)
# Faster models get higher reward, R_carbon ∈ [0, 1]
```

**Why this approach:**
- CodeCarbon provides absolute numbers for publication (accounts for GPU power, grid intensity)
- Relative efficiency provides stable reward signal for LA convergence
- Inference time is deterministic; carbon measurement has variance

**Implementation:**
- [ ] `src/la_framework/carbon_tracker.py` - CodeCarbon wrapper + relative scoring
- [ ] Profile inference time per model
- [ ] Implement uncertainty estimation (MC Dropout or ensemble variance)
- [ ] Weight sensitivity analysis

---

## Phase 3: Component 2 - Context-Aware Adaptation (RQ4 Core)

### 3.1 Multi-Resolution Context Features
- [ ] **High-res (10m):** Spectral statistics from Sentinel-2
- [ ] **Mid-res (30m):** Bathymetry, Substrate features
- [ ] **Low-res (future):** SST, Tide (when available)
- [ ] Handle missing auxiliary data with masking

### 3.2 Memory Bank
- [ ] FAISS-based k-NN retrieval
- [ ] Store: (context_vector, heuristic_probs, reward)
- [ ] Global-local blending with adaptive α

### 3.3 Cross-Sensor Context
- [ ] Extract comparable features from both Sentinel-2 and Landsat
- [ ] Test if context similarity works across sensors

---

## Phase 4: Component 3 - Sparse Label Handling & Non-Stationary Environments

### 4.1 Theoretical Foundation: Dual-Layer Non-Stationarity

**Key Insight:** Kelp forest monitoring exhibits TWO distinct sources of non-stationarity that must be handled by the LA framework. This dual-layer model extends classical LA theory on Non-Stationary Environments (NSEs).

#### Understanding MSE and PSE in LA Theory

**From Classical LA Theory (Narendra & Thathachar, Thathachar & Sastry):**

In a **Markovian Switching Environment (MSE)**, the automaton operates in a finite number of environments {E1, E2, ..., Ed} which are states of a Markov chain. The penalty probabilities change according to stochastic state transitions. The average penalty is:

```
M(n) = (1/T) Σᵢ pᵢ(n) [Σⱼ qⱼ(n) cⱼᵢ]
```

where qⱼ(n) is the probability of being in environment Eⱼ, and cⱼᵢ is the penalty probability for action αᵢ in environment Eⱼ.

In a **Periodic Switching Environment (PSE)**, the system evolves through k states according to a round-robin schedule with period T. The environment changes from state Qᵢ to Qᵢ₊₁ mod k after T discrete events.

**Critical Distinction:** Classical MSE/PSE describe how **penalty probabilities** (which action is optimal) change over time, NOT when feedback arrives.

#### Dual-Layer Model for Kelp Monitoring

We extend classical LA theory to model TWO layers of non-stationarity:

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: ENVIRONMENT DYNAMICS (Classical MSE/PSE)               │
│ What changes: Which heuristic is optimal for kelp detection     │
│ PSE: Seasonal variation (water conditions, kelp phenology)      │
│ MSE: Inter-annual climate regime shifts (El Niño, heatwaves)    │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2: FEEDBACK DYNAMICS (Novel Extension)                    │
│ What changes: When validation labels arrive                     │
│ PSE: Seasonal fieldwork calendar (summer peak, winter minimal)  │
│ MSE: Operational variability (funding, logistics, weather)      │
└─────────────────────────────────────────────────────────────────┘
```

**Scientific Contribution:** This dual-layer model is a novel extension of LA theory. Most NSE work focuses on Layer 1 (environment dynamics). Modeling Layer 2 (feedback timing) as a separate MSE/PSE process enables analysis of learning under both changing optimality AND irregular observation.

#### BC Data Temporal Coverage Assessment

**Current BC Sentinel-2 Data:**

| Date | Site | Season | Year |
|------|------|--------|------|
| July 27, 2021 | T09UXQ | Summer | 2021 |
| August 6, 2022 | T09UYQ | Summer | 2022 |
| August 6, 2022 | T10UCU | Summer | 2022 |
| August 4, 2023 | T09UXS | Summer | 2023 |
| August 16, 2023 | T10UCA | Summer | 2023 |
| August 19, 2023 | T09UWT | Summer | 2023 |
| September 2, 2023 | T09UUU | Summer | 2023 |

**Implications:**
- **Layer 1 (Environment Dynamics):** Cannot be empirically tested - no multi-temporal observations per site, all images from summer
- **Layer 2 (Feedback Dynamics):** Can be fully simulated and tested

**Approach:**
1. Layer 1: Simulate environment dynamics; use temporal metadata in context features
2. Layer 2: Implement full PSE+MSE feedback timing simulation
3. Future work: Acquire multi-temporal data for empirical Layer 1 testing

### 4.2 Prediction Buffer
- [ ] FIFO buffer with hash map lookup
- [ ] Simulate delayed label arrival (1 day to 6 months)

### 4.3 Layer 1: Environment Dynamics Simulation

Models how the **optimal heuristic** changes over time due to seasonal and climatic factors.

**PSE Component - Seasonal Variation:**
- **k = 4 states:** Winter, Spring, Summer, Fall
- **T = 90 days** per state
- Different heuristics optimal in different seasons (e.g., turbidity-robust models in winter)

**MSE Component - Climate Regime Shifts:**
- **d = 3 states:** Normal, El Niño, Marine Heatwave
- Transition probabilities based on climate dynamics
- Regime shifts change which heuristic performs best

```python
class EnvironmentDynamics:
    """Layer 1: Models which heuristic is optimal over time"""

    def __init__(self, heuristic_pool: List[str]):
        self.heuristics = heuristic_pool
        self.n_heuristics = len(heuristic_pool)

        # PSE: Seasonal performance modifiers per heuristic
        # Higher value = better performance in that season
        self.seasonal_strengths = {
            "winter": np.array([0.7, 0.9, 0.6, ...]),  # per heuristic
            "spring": np.array([0.8, 0.8, 0.7, ...]),
            "summer": np.array([0.9, 0.7, 0.9, ...]),
            "fall": np.array([0.8, 0.8, 0.8, ...]),
        }

        # MSE: Climate state transition matrix
        self.climate_P = np.array([
            [0.85, 0.10, 0.05],  # Normal → Normal, El Niño, Heatwave
            [0.30, 0.60, 0.10],  # El Niño → ...
            [0.40, 0.10, 0.50],  # Heatwave → ...
        ])
        self.climate_state = 0

        # Performance modifiers per climate state
        self.climate_modifiers = {
            0: np.ones(self.n_heuristics),        # Normal: baseline
            1: np.array([0.8, 1.1, 0.9, ...]),    # El Niño: some heuristics better
            2: np.array([0.7, 0.8, 1.2, ...]),    # Heatwave: different optimal
        }

    def get_optimal_heuristic(self, day_of_year: int) -> int:
        """Returns index of currently optimal heuristic"""
        season = self._get_season(day_of_year)
        seasonal_perf = self.seasonal_strengths[season]
        climate_perf = self.climate_modifiers[self.climate_state]
        combined_perf = seasonal_perf * climate_perf
        return np.argmax(combined_perf)

    def get_reward_modifier(self, heuristic_idx: int, day_of_year: int) -> float:
        """Returns performance modifier for given heuristic at given time"""
        season = self._get_season(day_of_year)
        seasonal = self.seasonal_strengths[season][heuristic_idx]
        climate = self.climate_modifiers[self.climate_state][heuristic_idx]
        return seasonal * climate

    def step_year(self):
        """MSE transition at year boundary"""
        self.climate_state = np.random.choice(3, p=self.climate_P[self.climate_state])
```

**Note:** With current BC data (single summer observation per site), Layer 1 effects are SIMULATED. The simulation provides:
- Theoretical validation of LA robustness to environment dynamics
- Framework for future empirical testing with multi-temporal data

### 4.4 Layer 2: Feedback Dynamics Simulation

Models **when validation labels arrive**, independent of environment dynamics.

#### PSE Component - Seasonal Fieldwork Calendar

```python
class PeriodicLabelArrival:
    """PSE-based seasonal label arrival"""
    def __init__(self):
        # Probability of label arriving per day in each season
        self.rates = {
            "winter": 0.02,   # Minimal fieldwork
            "spring": 0.15,   # Increasing activity
            "summer": 0.40,   # Peak field season
            "fall": 0.20      # Decreasing activity
        }
        self.T = 90  # Days per season

    def get_label_probability(self, day_of_year: int) -> float:
        seasons = ["winter", "spring", "summer", "fall"]
        season_idx = (day_of_year // self.T) % 4
        return self.rates[seasons[season_idx]]
```

#### MSE Component - Operational Variability

```python
class MarkovianPerturbation:
    """MSE-based stochastic inter-annual perturbation"""
    def __init__(self):
        # State transition matrix (year-to-year)
        self.P = np.array([
            [0.7, 0.2, 0.1],  # Normal → Normal, Reduced, Enhanced
            [0.4, 0.5, 0.1],  # Reduced → ...
            [0.3, 0.1, 0.6],  # Enhanced → ...
        ])
        self.multipliers = [1.0, 0.3, 2.0]  # Label rate multipliers
        self.state = 0  # Current state

    def step_year(self):
        self.state = np.random.choice(3, p=self.P[self.state])

    def get_multiplier(self) -> float:
        return self.multipliers[self.state]
```

#### Combined Realistic Feedback Model

```python
class RealisticLabelArrival:
    """Combines PSE (seasonal) + MSE (inter-annual) for feedback timing"""
    def __init__(self):
        self.pse = PeriodicLabelArrival()
        self.mse = MarkovianPerturbation()

    def step_year(self):
        self.mse.step_year()

    def label_arrives(self, day_of_year: int) -> bool:
        base_prob = self.pse.get_label_probability(day_of_year)
        adjusted_prob = base_prob * self.mse.get_multiplier()
        return np.random.random() < min(adjusted_prob, 1.0)

    def sample_delay(self, season: str) -> int:
        """Sample delay in days, conditioned on season"""
        if season == "summer":
            return int(np.random.lognormal(mean=2.0, sigma=0.5))  # ~7 days median
        elif season == "winter":
            return int(np.random.lognormal(mean=4.5, sigma=1.0))  # ~90 days median
        else:
            return int(np.random.lognormal(mean=3.5, sigma=0.8))  # ~30 days median
```

### 4.5 Scientific Justification

**Why This Dual-Layer Model?**

1. **Theoretical Grounding:** Both layers use established LA theory (MSE/PSE) but apply them to different aspects of the monitoring problem
2. **Domain Realism:** Kelp monitoring genuinely exhibits both:
   - Changing optimal detection strategies (seasonal phenology, climate regimes)
   - Irregular validation feedback (fieldwork constraints, operational variability)
3. **Novel Contribution:** Modeling feedback timing as a separate MSE/PSE process extends classical LA theory
4. **Ablation Capability:** The two layers can be independently controlled for rigorous experimental analysis

**Connection to Thesis Theory (Chapter 2):**
- PSE for feedback timing models the **phenological cycle** of kelp and **fieldwork calendar**
- MSE for feedback timing models **operational uncertainties** (funding, weather, logistics)
- PSE for environment dynamics models **seasonal variation** in detection challenges
- MSE for environment dynamics models **climate regime shifts** (El Niño, marine heatwaves)

### 4.6 Sparse Feedback Experiments

#### Ablation Matrix for Dual-Layer Non-Stationarity

```
─────────────────────────────────────────────────────────────────────────
Layer 1 (Environment)    Layer 2 (Feedback)       What We Test
─────────────────────────────────────────────────────────────────────────
Stationary               Immediate                Baseline (standard LA)
Stationary               PSE-only                 Seasonal feedback effect
Stationary               MSE-only                 Stochastic feedback effect
Stationary               PSE+MSE                  Full feedback dynamics
PSE-only                 Immediate                Seasonal environment effect
MSE-only                 Immediate                Climate regime effect
PSE+MSE                  Immediate                Full environment dynamics
PSE+MSE                  PSE+MSE                  Full realism (both layers)
─────────────────────────────────────────────────────────────────────────
```

- [ ] Label densities: 100%, 50%, 10%, 5%
- [ ] Layer 2 (Feedback): Immediate, Uniform, PSE-only, MSE-only, Combined
- [ ] Layer 1 (Environment): Stationary, PSE-only, MSE-only, Combined (simulated)
- [ ] Temporal decay analysis
- [ ] Cross-layer interaction effects

---

## Phase 5: Integration & LOSO Evaluation

### 5.1 Within-Sensor Evaluation (BC)
```
For each site S in [UWS, UXR, UXQ, UYQ, UCU, UXS, UCA, UWT, UDU, UUU]:
    Train on 9 sites, test on S
    Full system: Global + Context + Sparse
```

### 5.2 Cross-Sensor Evaluation
```
Train LA on BC Sentinel-2 (learn context-heuristic mappings)
Test context transfer on Figshare Landsat (using single RGB model)
```
- [ ] Measure domain gap in context feature space
- [ ] Test if memory bank similarity matching works across sensors
- [ ] Analyze context representation transfer quality

### 5.3 Ablation Studies (Staged Approach)

See **Phase 6: Rigorous Experimentation Plan** for full ablation protocol.

---

## Phase 6: Analysis & Publication

### 6.1 Rigorous Experimentation Plan

#### Staged Ablation Protocol

**Stage 1: Primary Experiments (Identify Best LA Variants)**
```yaml
config:
  la_variants: [LR-I, LR-P, VSLA, Pursuit, DiscretizedPursuit, SERI]
  components: [Global, Context, Sparse]  # Full system
  reward: single_objective  # IoU only
  sparsity: 1.0  # 100% labels
  delay: immediate
  eval: LOSO (10 sites)
output: Best 2-3 LA variants for subsequent stages
experiments: 6 variants × 10 sites = 60 runs
```

**Stage 2: Component Ablation (Test Contributions)**
```yaml
config:
  la_variants: [top 2-3 from Stage 1]
  components: [Global, Global+Context, Global+Sparse, Full]
  reward: single_objective
  sparsity: 1.0
  delay: immediate
  eval: LOSO (10 sites)
output: Component importance ranking
experiments: 3 variants × 4 combos × 10 sites = 120 runs
```

**Stage 3: Sparsity Ablation (Test RQ3)**
```yaml
config:
  la_variants: [best from Stage 2]
  components: [best config from Stage 2]
  reward: single_objective
  sparsity: [1.0, 0.5, 0.1, 0.05]
  delay: [immediate, uniform, pse, mse, combined]
  eval: LOSO (10 sites)
output: Degradation curves, sparsity tolerance
experiments: 1 × 4 sparsity × 5 delay × 10 sites = 200 runs
```

**Stage 4: Auxiliary Data Ablation (Test RQ4)**
```yaml
config:
  la_variants: [best from Stage 3]
  components: [best config]
  sparsity: [best tolerance from Stage 3]
  auxiliary: [none, bathymetry, substrate, both]
  eval: LOSO (10 sites)
output: Auxiliary data contribution
experiments: 1 × 4 auxiliary × 10 sites = 40 runs
```

**Stage 5: Cross-Sensor Evaluation**
```yaml
config:
  la_variants: [best from Stage 4]
  components: [best config]
  auxiliary: [best from Stage 4]
  train: BC Sentinel-2
  test: Figshare Landsat (RGB model)
output: Cross-sensor context transfer analysis
experiments: Focused evaluation set
```

**Total Estimated Experiments:** ~500 runs (vs 46,080 full permutation)

#### JSON Schema for Results Logging

All experiment results logged to `results/raw/phase_X/` in JSON format:

```json
{
  "experiment_id": "exp_20260314_001",
  "timestamp": "2026-03-14T10:30:00Z",
  "git_commit": "abc123def",
  "stage": 1,
  "config": {
    "la_variant": "LR-I",
    "components": ["global", "context", "sparse"],
    "reward": {
      "type": "single",
      "weights": {"iou": 1.0, "speed": 0.0, "uncertainty": 0.0, "carbon": 0.0}
    },
    "sparsity": {
      "label_density": 1.0,
      "delay_distribution": "immediate",
      "delay_params": {}
    },
    "auxiliary": ["bathymetry", "substrate"],
    "dataset": "bc_sentinel2",
    "eval_protocol": "loso",
    "test_site": "UWS",
    "seed": 42
  },
  "metrics": {
    "iou": {"mean": 0.847, "std": 0.023, "per_tile": [0.82, 0.85, ...]},
    "f1": {"mean": 0.881, "std": 0.019, "per_tile": [...]},
    "precision": {"mean": 0.902, "std": 0.021, "per_tile": [...]},
    "recall": {"mean": 0.861, "std": 0.025, "per_tile": [...]},
    "convergence": {
      "steps_to_90pct": 150,
      "steps_to_95pct": 280,
      "final_entropy": 0.312
    },
    "inference_ms": {"mean": 45.2, "std": 3.1, "per_heuristic": {...}},
    "carbon_kg": {"total": 0.00015, "per_tile_mean": 1.2e-7}
  },
  "dynamics": {
    "probability_evolution": {
      "steps": [0, 10, 20, 50, 100, 200, 500],
      "probabilities": {
        "h_UWS": [0.1, 0.12, 0.15, 0.22, 0.35, 0.42, 0.45],
        "h_UXR": [0.1, 0.11, 0.13, 0.18, 0.20, 0.22, 0.23]
      }
    },
    "reward_history": [0.72, 0.75, 0.78, 0.81, ...],
    "heuristic_selections": ["UWS", "UXR", "UWS", "UYQ", ...]
  },
  "per_tile_results": [
    {
      "tile_id": "tile_001",
      "context_vector": [0.45, 0.32, ...],
      "selected_heuristic": "UWS",
      "iou": 0.89,
      "f1": 0.91,
      "inference_ms": 44,
      "carbon_kg": 1.1e-7
    }
  ]
}
```

**Analysis Scripts:**
- `src/analysis/load_results.py` - Load JSON results into pandas DataFrames
- `src/analysis/aggregate_results.py` - Aggregate across experiments
- `src/analysis/statistical_tests.py` - Significance tests (paired t-test, Wilcoxon), effect sizes, CIs
- `src/analysis/generate_figures.py` - Create 4K figures with parameterized styles

### 6.2 Figure Generation

**Parameterized Style System:**

```python
# src/analysis/figure_config.py
FIGURE_STYLES = {
    "default": {
        "font_family": "sans-serif",
        "font_size": 10,
        "dpi": 400,
        "figsize": (7, 5),  # inches
        "colormap": "viridis",
        "grid": True,
        "spine_visible": ["left", "bottom"],
    },
    "nature": {
        "font_family": "Helvetica",
        "font_size": 8,
        "figsize": (3.5, 2.5),  # Nature single column
    },
    "rse": {  # Remote Sensing of Environment
        "font_family": "serif",
        "font_size": 9,
        "figsize": (6.5, 4.5),
    },
    "ieee": {
        "font_family": "Times New Roman",
        "font_size": 8,
        "figsize": (3.5, 2.625),
    },
}
```

**Output Formats:** Always save both PDF (vector) and PNG (raster at 400 DPI)

**Late Binding:** Style can be switched at publication time:
```bash
make figures STYLE=nature
make figures STYLE=rse
```

### 6.3 Figures for Each RQ
- **RQ1/RQ2:** LA convergence curves, LOSO heatmaps, probability evolution
- **RQ3:** Performance vs sparsity curves, delay distribution impact
- **RQ4:** Multi-resolution fusion analysis, cross-sensor transfer visualization

### 6.4 Statistical Analysis
- [ ] Significance tests (paired t-test, Wilcoxon signed-rank)
- [ ] Effect sizes (Cohen's d)
- [ ] Confidence intervals (95%)
- [ ] Multiple comparison correction (Bonferroni, FDR)

---

## Makefile for Research Automation

```makefile
# =============================================================================
# LA Kelp Detection Research - Makefile
# =============================================================================

PYTHON := python
STYLE := default
DPI := 400

# ========================= DATA =========================
data/bc_sentinel2:
	dvc pull data.dvc

data/figshare_landsat:
	$(PYTHON) src/preprocessing/download_figshare.py

data-prep: data/bc_sentinel2 data/figshare_landsat
	$(PYTHON) src/preprocessing/prepare_all.py

# ========================= EDA =========================
eda: data-prep
	jupyter nbconvert --execute src/notebooks/data_eda.ipynb --to html

figures-4k-samples:
	$(PYTHON) src/visualization/generate_sample_maps.py --dpi $(DPI) --output results/raw/phase_0_eda/sample_maps/

# ========================= MODELS =========================
models/rgb/rgb_all_sites.pt: data-prep
	$(PYTHON) src/training/train_rgb_model.py --output models/rgb/rgb_all_sites.pt

train-rgb: models/rgb/rgb_all_sites.pt

deploy-runpod:
	./scripts/runpod/deploy.sh

# ========================= EXPERIMENTS =========================
# Phase 2: Global Policy Learning
exp-phase2:
	$(PYTHON) src/experiments/run_global_la.py --config configs/phase2.yaml

# Phase 3: Context-Aware Adaptation
exp-phase3:
	$(PYTHON) src/experiments/run_context_la.py --config configs/phase3.yaml

# Phase 4: Sparse Label Handling
exp-phase4:
	$(PYTHON) src/experiments/run_sparse_la.py --config configs/phase4.yaml

# Phase 5: Integration
exp-phase5-loso:
	$(PYTHON) src/experiments/run_loso.py --config configs/phase5_loso.yaml

exp-phase5-cross-sensor:
	$(PYTHON) src/experiments/run_cross_sensor.py --config configs/phase5_cross.yaml

# Staged Ablations
exp-stage1:
	$(PYTHON) src/experiments/run_staged_ablation.py --stage 1 --config configs/ablation_stage1.yaml

exp-stage2:
	$(PYTHON) src/experiments/run_staged_ablation.py --stage 2 --config configs/ablation_stage2.yaml

exp-stage3:
	$(PYTHON) src/experiments/run_staged_ablation.py --stage 3 --config configs/ablation_stage3.yaml

exp-stage4:
	$(PYTHON) src/experiments/run_staged_ablation.py --stage 4 --config configs/ablation_stage4.yaml

exp-stage5:
	$(PYTHON) src/experiments/run_staged_ablation.py --stage 5 --config configs/ablation_stage5.yaml

exp-all-stages: exp-stage1 exp-stage2 exp-stage3 exp-stage4 exp-stage5

# ========================= ANALYSIS =========================
aggregate:
	$(PYTHON) src/analysis/aggregate_results.py --input results/raw/ --output results/aggregated/

stats:
	$(PYTHON) src/analysis/statistical_tests.py --input results/aggregated/

figures:
	$(PYTHON) src/analysis/generate_figures.py --style $(STYLE) --dpi $(DPI) --format pdf,png

analyze: aggregate stats figures

# ========================= FULL PIPELINE =========================
all: data-prep eda train-rgb exp-all-stages analyze

# ========================= UTILITIES =========================
clean-results:
	rm -rf results/raw/*

clean-figures:
	rm -rf results/papers/*/figures/*

lint:
	ruff check src/

test:
	pytest tests/ -v

.PHONY: all data-prep eda train-rgb deploy-runpod exp-all-stages analyze clean-results clean-figures lint test
```

---

## Compute Strategy

| Phase | Compute | Rationale |
|-------|---------|-----------|
| 0 (EDA) | Local M2 | Interactive exploration |
| 1 (Models) | RunPod | GPU training for RGB model |
| 2-4 (Components) | RunPod | Full GPU experiments |
| 5 (LOSO) | RunPod | 10-fold × multiple configs |
| 6 (Analysis) | Local M2 | Plotting, writing |

---

## Project Structure

```
thesis/                               # /Users/ebisong/Documents/code/uvic/thesis
├── data/
│   ├── bc_sentinel2/             # BC Sentinel-2 tiles
│   └── figshare_landsat/         # Cross-sensor dataset
├── models/
│   ├── site_specific/            # 10 LOO models (12-band)
│   ├── rgb/                      # Single RGB model for cross-sensor
│   ├── auxiliary/                # Models with Bathy/Substrate
│   └── experimental/             # Other configs
├── src/
│   ├── la_framework/             # LA algorithms
│   │   ├── automaton.py
│   │   ├── reward.py
│   │   ├── memory_bank.py
│   │   ├── sparse_handler.py
│   │   └── carbon_tracker.py
│   ├── heuristics/               # Model wrappers
│   │   └── dl/
│   │       └── heuristic_pool.py
│   ├── preprocessing/            # Data loaders
│   ├── training/                 # Model training scripts
│   │   └── train_rgb_model.py
│   ├── context/                  # Feature extraction
│   │   └── feature_extractor.py
│   ├── fusion/                   # RQ4 fusion code
│   │   ├── multi_resolution.py
│   │   └── cross_sensor.py
│   ├── visualization/            # Figure generation
│   │   └── generate_sample_maps.py
│   ├── analysis/                 # Results analysis
│   │   ├── load_results.py
│   │   ├── aggregate_results.py
│   │   ├── statistical_tests.py
│   │   ├── generate_figures.py
│   │   └── figure_config.py
│   ├── notebooks/                # EDA notebooks
│   │   └── data_eda.ipynb
│   └── experiments/              # Experiment runners
│       ├── run_global_la.py
│       ├── run_context_la.py
│       ├── run_sparse_la.py
│       ├── run_loso.py
│       ├── run_cross_sensor.py
│       └── run_staged_ablation.py
├── configs/                      # Experiment configurations
│   ├── phase2.yaml
│   ├── phase3.yaml
│   ├── phase4.yaml
│   ├── phase5_loso.yaml
│   ├── phase5_cross.yaml
│   └── ablation_stage*.yaml
├── scripts/
│   └── runpod/
│       ├── deploy.sh
│       ├── inference_server.py
│       ├── la_inference_pipeline.py
│       ├── batch_download.py
│       └── delayed_update.py
├── results/
│   ├── raw/                      # By experimental phase
│   │   ├── phase_0_eda/
│   │   │   └── sample_maps/      # 4K publication-ready maps
│   │   ├── phase_1_heuristics/
│   │   ├── phase_2_global_la/
│   │   ├── phase_3_context/
│   │   ├── phase_4_sparse/
│   │   ├── phase_5_loso/
│   │   └── phase_5_cross_sensor/
│   ├── aggregated/               # Processed results
│   └── papers/                   # Publication-ready
│       ├── paper_1_core_method/
│       ├── paper_2_sparse_learning/
│       ├── paper_3_cross_sensor/
│       ├── paper_4_bc_kelp/
│       └── paper_5_synthesis/
├── tests/                        # Unit tests
├── docs/
├── Makefile                      # Research automation
├── requirements.txt
├── CLAUDE.md                     # Project context
├── plan.md                       # This file
└── papers.md                     # Publication plan
```

---

## Results Structure

```
results/
├── raw/                              # JSON dumps per experiment
│   ├── phase_0_eda/
│   │   └── sample_maps/              # 4K PNG + PDF maps
│   ├── phase_1_heuristics/
│   ├── phase_2_global_la/
│   │   └── exp_*.json
│   ├── phase_3_context/
│   ├── phase_4_sparse/
│   ├── phase_5_loso/
│   └── phase_5_cross_sensor/
│
├── aggregated/                       # Pandas-ready CSVs
│   ├── stage1_summary.csv
│   ├── stage2_summary.csv
│   └── ...
│
└── papers/                           # Publication-ready
    ├── paper_1_core_method/
    │   ├── figures/                  # 4K PNG + PDF
    │   ├── tables/                   # LaTeX tables
    │   └── analysis/                 # Statistical outputs
    └── ...
```

**Workflow:**
1. Run experiments → JSON outputs to `results/raw/phase_X/`
2. Aggregate → CSVs to `results/aggregated/`
3. Generate figures → `results/papers/paper_Y/figures/`

---

## Critical Files

### Data
- `data/bc_sentinel2/` - BC Sentinel-2 tiles with masks
- `data/figshare_landsat/` - Figshare Landsat cross-sensor data

### Models
- `models/site_specific/` - 10 LOO site-specific models (12-band)
- `models/rgb/rgb_all_sites.pt` - Single RGB model for cross-sensor
- `models/auxiliary/` - Models with Bathy/Substrate

### Code (Existing to Modify)
- `src/la_framework/automaton.py`
- `src/la_framework/reward.py`
- `src/preprocessing/data_loader.py`

### Code (New to Create)
- `src/heuristics/dl/heuristic_pool.py` - Unified wrapper interface
- `src/la_framework/memory_bank.py` - FAISS-based k-NN
- `src/la_framework/sparse_handler.py` - PSE + MSE simulation
- `src/la_framework/carbon_tracker.py` - CodeCarbon + relative scoring
- `src/training/train_rgb_model.py` - RGB model training
- `src/context/feature_extractor.py` - Multi-resolution features
- `src/fusion/multi_resolution.py` - RQ4 fusion
- `src/fusion/cross_sensor.py` - Cross-sensor features
- `src/visualization/generate_sample_maps.py` - 4K map generation
- `src/analysis/*.py` - Analysis pipeline
- `src/experiments/*.py` - Experiment runners
- `src/notebooks/data_eda.ipynb` - EDA notebook

---

## Heuristic Pool Summary

### Primary Experiments (BC, 12-band)
```
H_12band = {h_UWS, h_UXR, h_UXQ, h_UYQ, h_UCU, h_UXS, h_UCA, h_UWT, h_UDU, h_UUU}
```
- 10 site-specific UNet-MaxViT models from `models/site_specific/`
- Each trained leaving one site out (LOO protocol)

### Cross-Sensor Experiments (RGB)
```
H_RGB = {h_RGB}  # Single model trained on ALL BC sites
```
- Single 3-band model: `models/rgb/rgb_all_sites.pt`
- Tests context transfer to Figshare Landsat

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Heuristics | 10 site-specific DL models (12-band) + 1 RGB model | Core selection on BC, context transfer on cross-sensor |
| RGB Training | Single model on all BC sites | Research goal is context transfer, not heuristic selection on Figshare |
| Deployment | Hybrid batch/sequential on RunPod | Matches operational monitoring workflow |
| Carbon | CodeCarbon for measurement, relative time for reward | Accurate publication numbers + stable LA training |
| Ablations | Staged approach (5 stages) | Efficient use of compute, ~500 vs 46,080 runs |
| **Non-Stationarity** | **Dual-layer MSE/PSE model** | **Layer 1: Environment dynamics (which heuristic optimal); Layer 2: Feedback timing (when labels arrive). Novel extension of LA theory.** |
| **Multi-Temporal Data** | **Download unlabeled Sentinel-2 for all 7 sites across seasons** | **Enables empirical Layer 1 testing; creates realistic sparse label scenario (~7% labeled)** |
| Layer 1 Testing | Empirical (with multi-temporal data) | Download seasonal imagery; evaluate via proxy metrics (coherence, plausibility) |
| Layer 2 Testing | Full implementation | PSE (seasonal fieldwork) + MSE (operational variability) |
| Preprocessing | Custom pipeline (matching Mohsen's format) | No access to original pipeline; implement GEE download + preprocessing |
| Figures | Parameterized styles, late binding | Flexible for different publication venues |
| Automation | Makefile | Reproducibility, straightforward repetitive execution |

---

## Open Items for Future

1. **Additional auxiliary data:** SST, tide height, chlorophyll (conversations ongoing)
2. **Falkland Islands data:** Contact Floating Forests researchers
3. **Hakai Institute data:** BC species-specific labels
4. **RQ5 (TEK):** Not in current scope, future work
