# Phase 1: Model Integration & Baseline

**Status:** 🔄 IN PROGRESS
**Prerequisite:** Phase 0 (Complete), Phase 0.5 (Optional)
**Last Updated:** March 19, 2026

---

## Objective

Establish the foundation for LA framework experiments by:
1. Creating a unified interface to all detection heuristics
2. Training a cross-sensor RGB model for transfer experiments
3. Setting up GPU infrastructure for experiments
4. Establishing baseline performance metrics

---

## Overview

Phase 1 bridges data preparation (Phase 0) and LA framework experiments (Phases 2-5). The core deliverables are:

| Deliverable | Purpose | Files | Status |
|-------------|---------|-------|--------|
| HeuristicPool | Unified interface to 10+ detection methods | `src/heuristics/dl/heuristic_pool.py` | ✅ COMPLETE |
| RGB Model | Cross-sensor transfer experiments | `models/rgb/rgb_all_sites.pt` | ⏸️ DEFERRED |
| RunPod Setup | GPU infrastructure for training/inference | `scripts/runpod/` | 🔲 TODO |
| Baselines | Performance benchmarks for comparison | `results/raw/phase_1_heuristics/` | 🔲 TODO |

> **Note:** RGB model training (Task 1.2) is deferred to a later phase. Cross-sensor experiments will be conducted after core LA framework validation on BC data.

---

## Task 1.1: Heuristic Pool Wrapper ✅ COMPLETE

**Completed:** March 19, 2026

### What is a HeuristicPool?

A software abstraction layer that provides a **unified interface** to all detection heuristics (models). This allows the LA framework to call any heuristic uniformly without knowing implementation details.

### Why is this needed?

The LA framework needs to:
1. Query available heuristics
2. Run inference with any heuristic
3. Compare outputs across heuristics
4. Track per-heuristic performance

Without a unified interface, the LA code would need heuristic-specific logic, making it brittle and hard to extend.

### Implementation

```python
# src/heuristics/dl/heuristic_pool.py

from typing import Dict, List, Optional
import torch
import numpy as np

class HeuristicPool:
    """
    Unified interface for all detection heuristics.

    The LA framework interacts with this class, not individual models.
    This enables easy addition of new heuristics without LA code changes.
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        """
        Load all available heuristics from model directory.

        Args:
            model_dir: Directory containing model checkpoints
            device: "cuda" or "cpu"
        """
        self.device = device
        self.models: Dict[str, torch.nn.Module] = {}
        self.metadata: Dict[str, dict] = {}

        # Load site-specific 12-band models
        for site in SITES:
            model_path = f"{model_dir}/site_specific/{site}.pt"
            self.models[f"12band_{site}"] = self._load_model(model_path)
            self.metadata[f"12band_{site}"] = {
                "input_channels": 12,
                "trained_site": site,
                "type": "site_specific"
            }

        # Load RGB model (for cross-sensor)
        rgb_path = f"{model_dir}/rgb/rgb_all_sites.pt"
        self.models["rgb_all"] = self._load_model(rgb_path)
        self.metadata["rgb_all"] = {
            "input_channels": 3,
            "trained_site": "all",
            "type": "cross_sensor"
        }

    def predict(self,
                tile: np.ndarray,
                heuristic_id: str,
                threshold: float = 0.5) -> np.ndarray:
        """
        Run inference with specified heuristic.

        This is the main interface used by LA framework.

        Args:
            tile: Input tile (C, H, W) numpy array
            heuristic_id: Which heuristic to use
            threshold: Binarization threshold

        Returns:
            Binary segmentation mask (H, W)
        """
        model = self.models[heuristic_id]

        # Preprocess
        tensor = self._preprocess(tile, heuristic_id)

        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)

        # Post-process
        mask = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        return mask

    def predict_proba(self,
                      tile: np.ndarray,
                      heuristic_id: str) -> np.ndarray:
        """
        Return probability map instead of binary mask.

        Useful for uncertainty estimation and soft voting.
        """
        model = self.models[heuristic_id]
        tensor = self._preprocess(tile, heuristic_id)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)

        return probs.squeeze().cpu().numpy()

    def list_heuristics(self,
                        filter_type: Optional[str] = None) -> List[str]:
        """
        List available heuristics.

        Args:
            filter_type: Optional filter ("site_specific", "cross_sensor")

        Returns:
            List of heuristic IDs
        """
        if filter_type is None:
            return list(self.models.keys())
        return [k for k, v in self.metadata.items()
                if v["type"] == filter_type]

    def get_metadata(self, heuristic_id: str) -> dict:
        """Get metadata for a specific heuristic."""
        return self.metadata[heuristic_id]

    def _preprocess(self, tile: np.ndarray, heuristic_id: str) -> torch.Tensor:
        """Preprocess tile for specific heuristic."""
        meta = self.metadata[heuristic_id]

        if meta["input_channels"] == 3:
            # RGB model: extract bands 4, 3, 2 (indices 2, 1, 0)
            tile = tile[[2, 1, 0], :, :]

        tensor = torch.from_numpy(tile).float().unsqueeze(0)
        return tensor.to(self.device)

    def _load_model(self, path: str) -> torch.nn.Module:
        """Load model checkpoint."""
        model = torch.load(path, map_location=self.device)
        model.eval()
        return model
```

### Tasks

- [x] Implement `HeuristicPool` class with unified `predict()` interface
- [x] Load site-specific UNet-MaxViT models (10-band spectral input)
- [x] Support variable input channels (10-band and 12-band with auto-stripping)
- [x] Add `predict_proba()` for uncertainty estimation
- [x] Unit tests for all methods (21 tests passing)
- [x] Ensemble methods (mean, vote, weighted)
- [x] Integration with LA framework `LADetector`

### Deliverables

| File | Purpose | Status |
|------|---------|--------|
| `src/heuristics/dl/heuristic_pool.py` | Main implementation | ✅ Created |
| `src/heuristics/dl/__init__.py` | Module exports | ✅ Created |
| `tests/test_heuristic_pool.py` | Unit tests (21 tests) | ✅ Created |

### Implementation Notes

The actual implementation differs from the planned design in several ways:

1. **Input Channels:** Models expect 10 spectral bands (B2-B12, excluding Bathymetry/Substrate), not 12. The implementation auto-strips auxiliary bands if 12-channel input is provided.

2. **Model Format:** Models are UNet-MaxViT with `tu-maxvit_tiny_tf_512` encoder. State dicts have `model.` prefix that needs stripping.

3. **Extends HeuristicBase:** `DLModelHeuristic` extends the existing `HeuristicBase` abstract class, ensuring compatibility with the LA framework.

4. **Currently 2 Models:** Only UCA and UCU site models have .pth files. Other sites only have metrics.json files (model files may need to be uploaded).

5. **Ensemble Methods:** Added `predict_ensemble()` with mean, vote, and weighted averaging methods.

---

## Task 1.2: Cross-Sensor RGB Model Training ⏸️ DEFERRED

> **Status:** Deferred to later phase. Focus first on LA framework validation using existing 12-band site-specific models on BC data. Cross-sensor experiments will follow after core framework is validated.

### Why a Single RGB Model?

The research goal is to test if LA's **context understanding transfers** across sensors, not to compare 12-band vs RGB performance. A single RGB model trained on all BC sites provides:

1. **Baseline for cross-sensor experiments** - Test on Figshare Landsat
2. **Simpler experimental design** - One model, clear interpretation
3. **Focus on context transfer** - Does memory bank similarity work across sensors?

### Training Protocol

| Aspect | Specification |
|--------|---------------|
| Architecture | UNet-MaxViT, `in_channels=3` |
| Input | BC Sentinel-2 bands 4, 3, 2 (RGB composite) |
| Training Split | 80/10/10 train/val/test across ALL BC tiles |
| Loss | Dice Loss + Binary Cross-Entropy (combined) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR or ReduceLROnPlateau |
| Epochs | 100 with early stopping (patience=10) |
| Augmentation | HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter |
| Batch Size | 8-16 (GPU memory dependent) |
| Validation Metric | IoU on validation set |

### Implementation

```python
# src/training/train_rgb_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from src.preprocessing.data_loader import BCKelpDataset
from src.models.unet_maxvit import UNetMaxViT

def train_rgb_model(config: dict):
    """
    Train single RGB model on all BC sites.

    This model will be used for:
    1. Cross-sensor testing on Figshare Landsat
    2. RGB baseline within BC dataset
    """

    # Data
    train_dataset = BCKelpDataset(
        data_dir=config["data_dir"],
        sites=None,  # All sites
        rgb_only=True,  # Only use bands 4, 3, 2
        split="train",
        transform=get_train_augmentations()
    )

    val_dataset = BCKelpDataset(
        data_dir=config["data_dir"],
        sites=None,
        rgb_only=True,
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Model
    model = UNetMaxViT(in_channels=3, out_channels=1)
    model = model.to(config["device"])

    # Loss & Optimizer
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # Training loop
    best_iou = 0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            images, masks = batch
            images = images.to(config["device"])
            masks = masks.to(config["device"])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_iou = evaluate_iou(model, val_loader, config["device"])

        # Early stopping
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), config["output_path"])
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_iou={val_iou:.4f}")

    return best_iou
```

### Tasks (DEFERRED)

- [ ] ~~Implement `src/training/train_rgb_model.py`~~
- [ ] ~~Add RGB extraction to data loader (`rgb_only=True` mode)~~
- [ ] ~~Implement `DiceBCELoss` combined loss function~~
- [ ] ~~Implement data augmentation pipeline~~
- [ ] ~~Train model on RunPod GPU~~
- [ ] ~~Validate on BC RGB subset~~
- [ ] ~~Save to `models/rgb/rgb_all_sites.pt`~~

### Deliverables (DEFERRED)

| File | Purpose | Status |
|------|---------|--------|
| `src/training/train_rgb_model.py` | Training script | Deferred |
| `src/training/losses.py` | Loss functions (DiceBCE) | Deferred |
| `src/training/augmentations.py` | Data augmentation | Deferred |
| `models/rgb/rgb_all_sites.pt` | Trained model | Deferred |
| `results/raw/phase_1_heuristics/rgb_training_log.json` | Training metrics | Deferred |

---

## Task 1.3: RunPod Deployment

### Why RunPod?

- **Cost-effective GPU access** - Pay per hour
- **Flexible scaling** - Spin up/down as needed
- **Docker support** - Reproducible environments

### Deployment Strategy: Hybrid Batch/Sequential

Operational kelp monitoring workflow:

1. **Batch Download:** Fetch all new Sentinel-2 tiles for BC coast (arrives every 5 days)
2. **Sequential LA Processing:** For each tile:
   - Extract context features
   - LA selects heuristic based on `p_final(c)`
   - Run selected model inference
   - Store prediction in buffer (awaiting validation)
3. **Delayed Update:** When validation labels arrive, update LA probabilities

### Implementation

```bash
# scripts/runpod/deploy.sh

#!/bin/bash
# Deploy LA Kelp framework to RunPod

# Environment setup
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Download models
dvc pull models/

# Start inference server
python scripts/runpod/inference_server.py --port 8080
```

```python
# scripts/runpod/inference_server.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.heuristics.dl.heuristic_pool import HeuristicPool
from src.la_framework.automaton import LAFramework

app = FastAPI()
pool = HeuristicPool("models/")
la = LAFramework(pool)

class TileRequest(BaseModel):
    tile_data: list  # Flattened tile
    shape: list      # (C, H, W)

class PredictionResponse(BaseModel):
    mask: list
    selected_heuristic: str
    confidence: float

@app.post("/predict")
async def predict(request: TileRequest) -> PredictionResponse:
    tile = np.array(request.tile_data).reshape(request.shape)
    mask, heuristic, conf = la.predict(tile)
    return PredictionResponse(
        mask=mask.flatten().tolist(),
        selected_heuristic=heuristic,
        confidence=conf
    )
```

### Tasks

- [ ] Create `scripts/runpod/deploy.sh` - Environment setup
- [ ] Create `scripts/runpod/inference_server.py` - FastAPI endpoint
- [ ] Create `scripts/runpod/la_inference_pipeline.py` - Full pipeline
- [ ] Create `scripts/runpod/batch_download.py` - Imagery batches
- [ ] Create `scripts/runpod/delayed_update.py` - Label processing
- [ ] Create `Dockerfile` for reproducibility
- [ ] Test deployment on RunPod instance

### Deliverables

| File | Purpose |
|------|---------|
| `scripts/runpod/deploy.sh` | Environment setup |
| `scripts/runpod/inference_server.py` | FastAPI server |
| `scripts/runpod/la_inference_pipeline.py` | Full LA pipeline |
| `scripts/runpod/batch_download.py` | Download imagery |
| `scripts/runpod/delayed_update.py` | Process labels |
| `Dockerfile` | Container definition |

---

## Task 1.4: Baseline Experiments

### Purpose

Establish performance benchmarks before LA framework experiments:
1. **Per-model performance** - How does each heuristic perform?
2. **Random selection baseline** - Lower bound for LA
3. **Oracle upper bound** - Best possible (cheating) performance

### Experiments

| Experiment | Description | Metric |
|------------|-------------|--------|
| Per-Model LOO | Each model on its held-out site | IoU, F1 |
| Random Selection | Randomly select heuristic per tile | Mean IoU |
| Oracle | Select best heuristic per tile (post-hoc) | Upper bound IoU |
| RGB vs 12-band | Compare RGB model to site-specific | IoU difference |

### Implementation

```python
# src/experiments/baseline_experiments.py

def run_per_model_baseline(pool: HeuristicPool, dataset: BCKelpDataset):
    """
    Evaluate each model on held-out site.

    For site-specific model h_S, evaluate on tiles from site S.
    This measures how well the model generalizes to its target site.
    """
    results = {}

    for heuristic_id in pool.list_heuristics(filter_type="site_specific"):
        site = pool.get_metadata(heuristic_id)["trained_site"]

        # Get tiles from this site only
        site_dataset = BCKelpDataset(sites=[site], split="test")

        iou_scores = []
        for tile, mask in site_dataset:
            pred = pool.predict(tile, heuristic_id)
            iou = compute_iou(pred, mask)
            iou_scores.append(iou)

        results[heuristic_id] = {
            "site": site,
            "mean_iou": np.mean(iou_scores),
            "std_iou": np.std(iou_scores),
            "n_tiles": len(iou_scores)
        }

    return results

def run_random_baseline(pool: HeuristicPool, dataset: BCKelpDataset, n_runs: int = 10):
    """
    Random heuristic selection baseline.

    This is the lower bound - LA should outperform random selection.
    """
    heuristics = pool.list_heuristics(filter_type="site_specific")

    all_runs = []
    for run in range(n_runs):
        run_ious = []
        for tile, mask in dataset:
            # Random selection
            h = np.random.choice(heuristics)
            pred = pool.predict(tile, h)
            iou = compute_iou(pred, mask)
            run_ious.append(iou)
        all_runs.append(np.mean(run_ious))

    return {
        "mean_iou": np.mean(all_runs),
        "std_iou": np.std(all_runs),
        "n_runs": n_runs
    }

def run_oracle_baseline(pool: HeuristicPool, dataset: BCKelpDataset):
    """
    Oracle (best-per-tile) baseline.

    This is the upper bound - LA cannot exceed oracle performance.
    """
    heuristics = pool.list_heuristics(filter_type="site_specific")

    oracle_ious = []
    for tile, mask in dataset:
        # Try all heuristics, pick best
        best_iou = 0
        for h in heuristics:
            pred = pool.predict(tile, h)
            iou = compute_iou(pred, mask)
            best_iou = max(best_iou, iou)
        oracle_ious.append(best_iou)

    return {
        "mean_iou": np.mean(oracle_ious),
        "std_iou": np.std(oracle_ious),
        "description": "Upper bound: best heuristic selected per tile"
    }
```

### Tasks

- [ ] Implement per-model LOO evaluation
- [ ] Implement random selection baseline
- [ ] Implement oracle upper bound
- [ ] Compare RGB vs 12-band performance
- [ ] Generate baseline results JSON
- [ ] Create baseline comparison figure

### Deliverables

| File | Purpose |
|------|---------|
| `src/experiments/baseline_experiments.py` | Baseline runners |
| `results/raw/phase_1_heuristics/per_model_results.json` | Per-model IoU |
| `results/raw/phase_1_heuristics/random_baseline.json` | Random baseline |
| `results/raw/phase_1_heuristics/oracle_baseline.json` | Oracle upper bound |
| `results/raw/phase_1_heuristics/baseline_comparison.png` | Visualization |

---

## Validation Criteria

### HeuristicPool ✅
- [x] Available site-specific models load correctly (UCA, UCU)
- [ ] All 10 site-specific models load (pending model files upload)
- [ ] RGB model loads correctly (DEFERRED - part of Task 1.2)
- [x] `predict()` returns valid binary masks (512x512, float32, 0/1 values)
- [x] `predict_proba()` returns values in [0, 1]
- [x] Unit tests pass (21 tests)
- [x] Integration with LADetector works

### RGB Model (DEFERRED)
- [ ] ~~Training converges (loss decreases)~~
- [ ] ~~Validation IoU > 0.5 (reasonable performance)~~
- [ ] ~~Model saves and loads correctly~~
- [ ] ~~Inference runs on both GPU and CPU~~

### RunPod
- [ ] Deploy script completes without errors
- [ ] Inference server responds to requests
- [ ] GPU utilization during inference

### Baselines
- [ ] Oracle > Random (sanity check)
- [ ] Per-model results are reasonable (IoU 0.3-0.9)
- [ ] Results logged in JSON format

---

## Dependencies

### From Phase 0
- `data/bc_sentinel2/` - Tiles and masks
- `data/figshare_landsat/` - Cross-sensor data with masks
- `src/preprocessing/data_loader.py` - Data loaders

### External
- 10 site-specific UNet-MaxViT models (from Mohsen)
- RunPod account and credits
- CUDA-capable GPU

---

## Compute Requirements

| Task | Compute | Estimated Time | Status |
|------|---------|----------------|--------|
| HeuristicPool | Local M2 | 1-2 hours | TODO |
| RGB Training | RunPod A100 | 4-8 hours | DEFERRED |
| RunPod Setup | RunPod | 1-2 hours | TODO |
| Baselines | RunPod | 2-4 hours | TODO |

---

## Next Steps After Phase 1

Upon completion of Phase 1, the following are ready:
1. **HeuristicPool** - LA framework can query and use any heuristic
2. **RunPod** - GPU infrastructure for Phase 2-5 experiments
3. **Baselines** - Reference points for LA performance

> **Deferred:** RGB model training will be done later for cross-sensor experiments.

**Phase 2** (Global Policy Learning) can then begin, implementing the LA automaton and testing different LA variants (LR-I, LR-P, VSLA, etc.) on the heuristic pool using BC 12-band data.
