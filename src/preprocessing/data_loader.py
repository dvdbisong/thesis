"""
Data Loader for Sentinel-2 Kelp Detection

Loads tile/mask pairs from the data_bc directory structure.
Provides PyTorch Dataset interface for training and evaluation.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader


class KelpTileDataset(Dataset):
    """
    PyTorch Dataset for loading Sentinel-2 kelp detection tiles.

    Directory structure expected:
        data_bc/
        ├── Tiles/
        │   └── {scene_id}/
        │       ├── images/
        │       │   └── tile_{N}_image.tiff
        │       └── masks/
        │           └── tile_{N}_mask.tiff
    """

    # Sentinel-2 band indices (10 bands used for spectral analysis)
    # The data has 12 bands total, last 2 are likely substrate/bathymetry
    BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    SPECTRAL_BANDS = 10  # First 10 bands are spectral

    def __init__(
        self,
        data_dir: str,
        tile_paths: Optional[List[Dict[str, str]]] = None,
        normalize: bool = True,
        bands: Optional[List[int]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to data_bc directory
            tile_paths: Optional list of tile path dictionaries. If None, discovers all tiles.
            normalize: Whether to normalize spectral values to [0, 1]
            bands: Optional list of band indices to load (0-indexed). If None, loads first 10.
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.bands = bands if bands is not None else list(range(self.SPECTRAL_BANDS))

        if tile_paths is not None:
            self.tile_paths = tile_paths
        else:
            self.tile_paths = self._discover_tiles()

        print(f"Loaded {len(self.tile_paths)} tiles from {self.data_dir}")

    def _discover_tiles(self) -> List[Dict[str, str]]:
        """Discover all tile/mask pairs in the data directory."""
        tiles = []
        tiles_dir = self.data_dir / "Tiles"

        if not tiles_dir.exists():
            raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

        for scene_dir in sorted(tiles_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("."):
                continue

            scene_id = scene_dir.name
            images_dir = scene_dir / "images"
            masks_dir = scene_dir / "masks"

            if not images_dir.exists() or not masks_dir.exists():
                continue

            for image_file in sorted(images_dir.glob("tile_*_image.tiff")):
                tile_num = image_file.stem.replace("_image", "").replace("tile_", "")
                mask_file = masks_dir / f"tile_{tile_num}_mask.tiff"

                if mask_file.exists():
                    tiles.append({
                        "image_path": str(image_file),
                        "mask_path": str(mask_file),
                        "scene_id": scene_id,
                        "tile_id": f"{scene_id}_tile_{tile_num}",
                    })

        return tiles

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a tile/mask pair.

        Args:
            idx: Index of the tile

        Returns:
            Tuple of (image_tensor, mask_tensor, metadata)
            - image_tensor: Shape (C, H, W) with selected bands
            - mask_tensor: Shape (H, W) binary mask
            - metadata: Dictionary with tile_id, scene_id, etc.
        """
        tile_info = self.tile_paths[idx]

        # Load image
        with rasterio.open(tile_info["image_path"]) as src:
            # Read selected bands (1-indexed in rasterio)
            bands_to_read = [b + 1 for b in self.bands]
            image = src.read(bands_to_read).astype(np.float32)

        # Load mask
        with rasterio.open(tile_info["mask_path"]) as src:
            mask = src.read(1).astype(np.float32)

        # Normalize if requested
        if self.normalize:
            # Clip to reasonable range and normalize to [0, 1]
            # Sentinel-2 reflectance values are typically 0-10000
            image = np.clip(image, 0, 10000) / 10000.0

        # Convert to tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        metadata = {
            "tile_id": tile_info["tile_id"],
            "scene_id": tile_info["scene_id"],
            "image_path": tile_info["image_path"],
            "mask_path": tile_info["mask_path"],
        }

        return image_tensor, mask_tensor, metadata

    def get_band_index(self, band_name: str) -> int:
        """
        Get the index of a band by name.

        Args:
            band_name: Band name (e.g., "B8", "B4")

        Returns:
            Index in the loaded tensor
        """
        if band_name not in self.BAND_NAMES:
            raise ValueError(f"Unknown band: {band_name}. Valid bands: {self.BAND_NAMES}")

        full_idx = self.BAND_NAMES.index(band_name)
        if full_idx not in self.bands:
            raise ValueError(f"Band {band_name} not loaded. Loaded bands: {self.bands}")

        return self.bands.index(full_idx)


def create_splits(
    data_dir: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratify_by_scene: bool = True,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/val/test splits.

    Args:
        data_dir: Path to data_bc directory
        split_ratio: Tuple of (train, val, test) ratios
        stratify_by_scene: If True, split by scene to test generalization
        seed: Random seed for reproducibility
        output_dir: Optional directory to save split JSON files

    Returns:
        Tuple of (train_tiles, val_tiles, test_tiles)
    """
    np.random.seed(seed)

    # Discover all tiles
    dataset = KelpTileDataset(data_dir, normalize=False)
    all_tiles = dataset.tile_paths

    if stratify_by_scene:
        # Group tiles by scene
        scenes = {}
        for tile in all_tiles:
            scene = tile["scene_id"]
            if scene not in scenes:
                scenes[scene] = []
            scenes[scene].append(tile)

        # Sort scenes by year (older for training, newer for testing)
        scene_names = sorted(scenes.keys())

        # Split scenes
        n_scenes = len(scene_names)
        n_train = int(n_scenes * split_ratio[0])
        n_val = int(n_scenes * split_ratio[1])

        train_scenes = scene_names[:n_train]
        val_scenes = scene_names[n_train : n_train + n_val]
        test_scenes = scene_names[n_train + n_val :]

        train_tiles = [t for s in train_scenes for t in scenes[s]]
        val_tiles = [t for s in val_scenes for t in scenes[s]]
        test_tiles = [t for s in test_scenes for t in scenes[s]]

        print(f"Train scenes: {train_scenes}")
        print(f"Val scenes: {val_scenes}")
        print(f"Test scenes: {test_scenes}")

    else:
        # Random split
        np.random.shuffle(all_tiles)
        n = len(all_tiles)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        train_tiles = all_tiles[:n_train]
        val_tiles = all_tiles[n_train : n_train + n_val]
        test_tiles = all_tiles[n_train + n_val :]

    print(f"Split: {len(train_tiles)} train, {len(val_tiles)} val, {len(test_tiles)} test")

    # Save splits if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, tiles in [("train", train_tiles), ("val", val_tiles), ("test", test_tiles)]:
            with open(output_path / f"{name}.json", "w") as f:
                json.dump(tiles, f, indent=2)

        print(f"Saved splits to {output_path}")

    return train_tiles, val_tiles, test_tiles


def load_splits(splits_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load existing splits from JSON files.

    Args:
        splits_dir: Directory containing train.json, val.json, test.json

    Returns:
        Tuple of (train_tiles, val_tiles, test_tiles)
    """
    splits_path = Path(splits_dir)

    train_tiles = json.load(open(splits_path / "train.json"))
    val_tiles = json.load(open(splits_path / "val.json"))
    test_tiles = json.load(open(splits_path / "test.json"))

    return train_tiles, val_tiles, test_tiles


def get_data_loaders(
    data_dir: str,
    splits_dir: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for train/val/test sets.

    Args:
        data_dir: Path to data_bc directory
        splits_dir: Path to splits directory. If None, creates new splits.
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for KelpTileDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if splits_dir and Path(splits_dir).exists():
        train_tiles, val_tiles, test_tiles = load_splits(splits_dir)
    else:
        train_tiles, val_tiles, test_tiles = create_splits(
            data_dir, output_dir=splits_dir
        )

    train_dataset = KelpTileDataset(data_dir, tile_paths=train_tiles, **dataset_kwargs)
    val_dataset = KelpTileDataset(data_dir, tile_paths=val_tiles, **dataset_kwargs)
    test_dataset = KelpTileDataset(data_dir, tile_paths=test_tiles, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data loader for kelp detection")
    parser.add_argument("--verify", action="store_true", help="Verify data integrity")
    parser.add_argument(
        "--data-dir",
        default="../data_bc",
        help="Path to data directory",
    )
    args = parser.parse_args()

    # Test the data loader
    print("Testing data loader...")
    dataset = KelpTileDataset(args.data_dir)

    if len(dataset) > 0:
        image, mask, metadata = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Metadata: {metadata}")
        print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
        print(f"Mask unique values: {torch.unique(mask)}")

        if args.verify:
            print("\nVerifying all tiles...")
            errors = []
            for i in range(len(dataset)):
                try:
                    img, msk, meta = dataset[i]
                    assert img.shape == (10, 512, 512), f"Bad image shape: {img.shape}"
                    assert msk.shape == (512, 512), f"Bad mask shape: {msk.shape}"
                except Exception as e:
                    errors.append((i, str(e)))

            if errors:
                print(f"Found {len(errors)} errors:")
                for idx, err in errors[:10]:
                    print(f"  Tile {idx}: {err}")
            else:
                print(f"All {len(dataset)} tiles verified successfully!")
