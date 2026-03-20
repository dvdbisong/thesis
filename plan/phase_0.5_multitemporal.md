# Phase 0.5: Multi-Temporal Data Acquisition & Preprocessing

**Status:** 🔄 IN PROGRESS
**Date Started:** March 20, 2026

---

## Objective

Download seasonal Sentinel-2 imagery for BC kelp sites across 2021-2023 to enable **empirical testing of the dual-layer MSE/PSE framework**. This creates a realistic sparse-label scenario where:
- ~7 labeled summer images (from Phase 0)
- ~80-100 unlabeled images across all seasons
- **~7% label density** - matches operational monitoring reality

### Scientific Rationale

The current BC data has single observations per site (all summer). This prevents empirical testing of **Layer 1 (Environment Dynamics)** in the dual-layer framework:

| Layer | What It Models | Current Status | With Multi-Temporal |
|-------|---------------|----------------|---------------------|
| Layer 1 | Which heuristic is optimal (seasonal/climate) | Cannot test | Empirical testing |
| Layer 2 | When labels arrive (fieldwork calendar) | Can simulate | Can simulate |

---

## Target Sites

| Site Code | Location | Labeled Date | Coordinates |
|-----------|----------|--------------|-------------|
| T09UUU | Haida Gwaii (Southern) | Sept 2023 | 52.97°N, 131.64°W |
| T09UWT | Central Coast (Bella Bella) | Aug 2023 | 51.94°N, 128.48°W |
| T09UWS | North Vancouver Island | Sept 2020 | 50.82°N, 127.62°W |
| T09UXS | Johnstone Strait | Aug 2023 | 50.53°N, 126.90°W |
| T10UCA | Discovery Islands | Aug 2023 | 50.23°N, 125.41°W |
| T09UXR | Nootka Sound | Sept 2020 | 49.82°N, 127.01°W |
| T09UXQ | Clayoquot Sound (Tofino) | July 2021 | 49.40°N, 126.53°W |
| T09UYQ | Barkley Sound (Ucluelet) | Aug 2022 | 48.76°N, 125.09°W |
| T10UCU | Southern Vancouver Island | Aug 2022 | 48.58°N, 124.60°W |
| T10UDU | Victoria / Juan de Fuca | Sept 2023 | 48.43°N, 124.01°W |

---

## Temporal Coverage

| Season | Months | Target Images/Site | Rationale |
|--------|--------|-------------------|-----------|
| Winter | Dec-Feb | 1-2 | Low kelp canopy, high turbidity |
| Spring | Mar-May | 2-3 | Rapid growth phase |
| Summer | Jun-Aug | 2-3 (includes labeled) | Peak canopy |
| Fall | Sep-Nov | 2-3 | Senescence onset |

**Years:** 2021, 2022, 2023

**Expected Volume:**
```
10 sites × 4 seasons × ~2 images/season × 3 years = ~240 candidates
After quality filtering (~40% usable): ~80-100 images
At ~30-40 tiles/image: ~3,000+ tiles
```

---

## Quality Criteria (Schroeder et al. 2020)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Cloud cover | < 20% | Metadata filter |
| Valid pixels | > 70% | After SCL masking |
| Sun elevation | > 30° | Avoid low-angle artifacts |

Quality score formula:
```
Q = 0.4 × (100 - cloud%) + 0.4 × valid% + 0.2 × sun_score
```

---

## Workflow

### Step 1: GEE Authentication

```bash
# One-time setup
pip install earthengine-api
earthengine authenticate
```

### Step 2: Dry Run (Query Without Download)

```bash
cd /Users/ebisong/Documents/code/uvic/thesis

# Check available imagery without downloading
python -m src.preprocessing.gee_download \
    --output data/bc_sentinel2_multitemporal/raw/ \
    --dry-run
```

### Step 3: Export Imagery to GCS

```bash
# Export all sites, all seasons, all years to Google Cloud Storage
python -m src.preprocessing.gee_download \
    --output data/bc_sentinel2_multitemporal/raw/ \
    --max-images 2

# Export specific site/season
python -m src.preprocessing.gee_download \
    --output data/bc_sentinel2_multitemporal/raw/ \
    --sites T09UXQ \
    --seasons summer fall \
    --years 2021 2022
```

**Note:** Exports go to GCS bucket `gs://uvic-thesis/kelp_multitemporal/`. Monitor task status in [GEE Task Manager](https://code.earthengine.google.com/tasks).

### Step 4: Download from GCS

```bash
# Check export status
python -m src.preprocessing.gee_download --check-status

# Download completed exports to local
python -m src.preprocessing.gee_download \
    --download-from-gcs \
    --output data/bc_sentinel2_multitemporal/raw/
```

**Note:** Requires Google Cloud SDK (`gsutil`) to be installed and authenticated.

### Step 5: Preprocessing

```bash
# Process downloaded imagery to match Mohsen's format
python -m src.preprocessing.preprocess_multitemporal \
    --input data/bc_sentinel2_multitemporal/raw/ \
    --output data/bc_sentinel2_multitemporal/Tiles/ \
    --auxiliary "data/bc_sentinel2/new/Masks 10 scenes/"
```

### Step 6: Validation

```bash
# Validate tile dimensions and band counts
python -m src.preprocessing.tile_creator validate \
    --dir data/bc_sentinel2_multitemporal/Tiles/{scene_id}/
```

---

## File Structure

```
data/bc_sentinel2_multitemporal/
├── raw/                              # Raw GEE downloads
│   ├── download_manifest.json        # Metadata for all downloads
│   └── {image_id}/
│       ├── {image_id}_B2B3B4B8.tif   # 10m bands
│       └── {image_id}_B5B6B7B8A_B11B12.tif  # 20m bands
│
├── processed/                        # Intermediate (stacked images)
│   └── {image_id}_{site}/
│       └── stacked_12band.tif
│
├── Tiles/                            # Final tiled output
│   └── {image_id}_{site}/
│       ├── images/
│       │   └── tile_{N}_image.tiff   # 12 bands, 512×512
│       └── masks/
│           └── tile_{N}_mask.tiff    # All zeros (no labels)
│
├── preprocessing_log.json            # Processing status
└── README.md                         # Documentation
```

---

## Proxy Evaluation Metrics

Since multi-temporal images have no ground truth, we use proxy metrics:

### 1. Temporal Coherence
```python
from src.evaluation import temporal_coherence

result = temporal_coherence(predictions, timestamps, window_days=30)
# High coherence = consistent predictions within time windows
```

### 2. Phenological Plausibility
```python
from src.evaluation import phenological_plausibility

result = phenological_plausibility(monthly_extents, expected_pattern="temperate")
# Score > 0.5 = matches expected summer peak, winter minimum
```

### 3. Heuristic Agreement
```python
from src.evaluation import heuristic_agreement

result = heuristic_agreement(predictions_by_heuristic)
# Low agreement = challenging conditions where LA adds value
```

### 4. LA Convergence
```python
from src.evaluation import la_convergence_metrics

result = la_convergence_metrics(probability_history, heuristic_names)
# Low entropy = confident heuristic selection
```

---

## Integration with Dual-Layer Framework

### Layer 1: Environment Dynamics (Empirical Testing)

```
Experimental Design:
────────────────────────────────────────────────────────────────
1. Train LA on labeled summer images (original 7)
2. Deploy LA on multi-temporal unlabeled images
3. Observe: Does LA change heuristic selection based on season?
4. Measure: Do seasonal selections match expected patterns?
   - Winter: Turbidity-robust heuristics selected more often?
   - Summer: Optimal-condition heuristics selected?
5. Evaluate: Proxy metrics (coherence, plausibility, agreement)
────────────────────────────────────────────────────────────────
```

### Layer 2: Feedback Dynamics (Simulation)

The multi-temporal data enables realistic feedback simulation:
- Labels arrive only for ~7% of tiles (summer only)
- PSE: Seasonal fieldwork calendar
- MSE: Operational variability

---

## Files Created

| File | Purpose |
|------|---------|
| `src/preprocessing/gee_download.py` | GEE download script |
| `src/preprocessing/preprocess_multitemporal.py` | Preprocessing pipeline |
| `src/preprocessing/tile_creator.py` | Tile creation/validation utility |
| `src/evaluation/__init__.py` | Evaluation module |
| `src/evaluation/unlabeled_metrics.py` | Proxy metrics for unlabeled data |

---

## Makefile Targets

```makefile
# Download multi-temporal imagery
download-multitemporal:
	python -m src.preprocessing.gee_download \
		--output data/bc_sentinel2_multitemporal/raw/

# Preprocess to tiles
preprocess-multitemporal:
	python -m src.preprocessing.preprocess_multitemporal \
		--input data/bc_sentinel2_multitemporal/raw/ \
		--output data/bc_sentinel2_multitemporal/Tiles/ \
		--auxiliary "data/bc_sentinel2/new/Masks 10 scenes/"

# Validate tiles
validate-multitemporal:
	python -m src.preprocessing.tile_creator validate \
		--dir data/bc_sentinel2_multitemporal/Tiles/

# Full pipeline
multitemporal-all: download-multitemporal preprocess-multitemporal validate-multitemporal
```

---

## Checklist

### Download Phase
- [ ] GEE authentication configured (`earthengine authenticate`)
- [ ] GCS authentication configured (`gcloud auth login`)
- [ ] Dry run completed successfully
- [ ] Full export initiated to GCS
- [ ] Exports completed in GEE Tasks
- [ ] Downloaded from GCS to local (`--download-from-gcs`)

### Preprocessing Phase
- [ ] Raw imagery preprocessed
- [ ] Auxiliary data aligned
- [ ] Tiles created (512×512, 12 bands)
- [ ] Empty masks created

### Validation Phase
- [ ] Tile dimensions verified
- [ ] Band counts verified
- [ ] No all-zero tiles
- [ ] Preprocessing log reviewed

### Integration Phase
- [ ] Data loader updated for multi-temporal
- [ ] Proxy metrics tested
- [ ] Ready for Layer 1 experiments

---

## Dependencies

```
earthengine-api>=0.1.370
geemap>=0.30.0
rasterio>=1.3.0
numpy>=1.24.0
```

---

## Notes

1. **GEE Quotas:** Be aware of GEE export quotas. Process in batches if needed.
2. **GCS Bucket:** Exports go to `gs://uvic-thesis/kelp_multitemporal/` (~50GB expected).
3. **Auxiliary Data:** Reused from existing labeled images (static over time).
4. **T10UDU:** May lack auxiliary data - handle gracefully in preprocessing.
5. **gsutil:** Required for downloading from GCS. Install via `pip install google-cloud-storage` or Google Cloud SDK.
