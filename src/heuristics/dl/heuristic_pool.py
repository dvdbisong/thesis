"""
Heuristic Pool for Deep Learning Models

Provides a unified interface to load and run site-specific deep learning models
as heuristics for the Learning Automata framework.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from src.heuristics.base import HeuristicBase


class DLModelHeuristic(HeuristicBase):
    """
    A deep learning model wrapped as a heuristic.

    Wraps a UNet-MaxViT segmentation model to conform to the HeuristicBase interface.
    Models are expected to have 10 input channels (Sentinel-2 spectral bands).
    """

    # Default input band order (10 spectral bands, excluding aux layers)
    DEFAULT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    def __init__(
        self,
        name: str,
        model_path: Path,
        device: Optional[str] = None,
        threshold: float = 0.5,
        input_channels: int = 10,
        **kwargs,
    ):
        """
        Initialize the DL model heuristic.

        Args:
            name: Unique identifier for this heuristic
            model_path: Path to the .pth model file
            device: Device to run inference on (None = auto-detect)
            threshold: Threshold for binary prediction (default 0.5)
            input_channels: Number of input channels (default 10)
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.input_channels = input_channels

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Load metrics if available
        self.metrics = self._load_metrics()

    def _load_model(self) -> nn.Module:
        """
        Load the segmentation model from the checkpoint.

        Returns:
            Loaded PyTorch model
        """
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required. "
                "Install with: pip install segmentation-models-pytorch"
            )

        # Create model architecture
        model = smp.Unet(
            encoder_name="tu-maxvit_tiny_tf_512",
            encoder_weights=None,  # We load from checkpoint
            in_channels=self.input_channels,
            classes=1,  # Binary segmentation
        )

        # Load state dict
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if "model" in state_dict or any(k.startswith("model.") for k in state_dict.keys()):
            # Checkpoint has 'model.' prefix - need to strip it
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                elif k not in ("mean", "std"):  # Skip normalization params
                    new_state_dict[k] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=True)
        return model

    def _load_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Load metrics JSON if it exists alongside the model.

        Returns:
            Metrics dictionary or None
        """
        metrics_path = self.model_path.with_name(
            self.model_path.stem + "_metrics.json"
        )
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return None

    @torch.no_grad()
    def predict(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Generate a binary prediction mask for the input tile.

        Args:
            tile: Input tensor of shape (C, H, W) with C spectral bands.
                  Expected to have 10 or 12 channels. If 12, auxiliary
                  bands (Bathymetry, Substrate) are stripped.

        Returns:
            Binary mask of shape (H, W) where 1=kelp, 0=no-kelp
        """
        # Handle input channels
        if tile.shape[0] == 12:
            # Strip auxiliary bands (last 2 channels: Substrate, Bathymetry)
            tile = tile[:10]
        elif tile.shape[0] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} or 12 input channels, "
                f"got {tile.shape[0]}"
            )

        # Add batch dimension and move to device
        x = tile.unsqueeze(0).to(self.device)

        # Normalize input (assuming raw reflectance values)
        # The models were trained on normalized data
        x = x.float()

        # Forward pass
        logits = self.model(x)

        # Apply sigmoid and threshold
        probs = torch.sigmoid(logits)
        mask = (probs > self.threshold).float()

        # Remove batch and channel dimensions
        mask = mask.squeeze(0).squeeze(0)

        return mask.cpu()

    def predict_proba(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Generate probability map (before thresholding).

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            Probability map of shape (H, W) with values in [0, 1]
        """
        # Handle input channels
        if tile.shape[0] == 12:
            tile = tile[:10]

        x = tile.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        return probs.squeeze(0).squeeze(0).cpu()

    def get_config(self) -> Dict[str, Any]:
        """Get configuration of this heuristic."""
        config = super().get_config()
        config.update({
            "model_path": str(self.model_path),
            "device": self.device,
            "threshold": self.threshold,
            "input_channels": self.input_channels,
        })
        return config


class HeuristicPool:
    """
    Pool of deep learning heuristics for the LA framework.

    Manages a collection of DL model heuristics, providing:
    - Automatic discovery and loading of models from a directory
    - Unified interface for LA framework integration
    - Metadata access for model selection strategies
    """

    # Regex to parse model filenames
    # Format: {InputConfig}_{Arch}_{Encoder}_{Size}_{Site}_{Date}.pth
    MODEL_PATTERN = re.compile(
        r"(?P<input_config>\w+)_"
        r"(?P<arch>Unet)_"
        r"(?P<encoder>tu-maxvit_tiny_tf_\d+)_"
        r"(?P<site>\w+)_"
        r"(?P<date>\d{8}_\d{6})"
        r"\.pth$"
    )

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        load_on_init: bool = True,
    ):
        """
        Initialize the heuristic pool.

        Args:
            models_dir: Directory containing model files. If None, uses
                        default location: models/site_specific/
            device: Device for inference (None = auto-detect)
            threshold: Default threshold for binary predictions
            load_on_init: Whether to load all models immediately
        """
        if models_dir is None:
            # Default to project models directory
            project_root = Path(__file__).parent.parent.parent.parent
            models_dir = project_root / "models" / "site_specific"

        self.models_dir = Path(models_dir)
        self.device = device
        self.threshold = threshold

        # Registry of available models
        self._model_registry: Dict[str, Dict[str, Any]] = {}

        # Loaded model heuristics
        self._heuristics: Dict[str, DLModelHeuristic] = {}

        # Discover available models
        self._discover_models()

        if load_on_init:
            self.load_all()

    def _discover_models(self) -> None:
        """
        Discover available model files in the models directory.
        """
        if not self.models_dir.exists():
            return

        for model_path in self.models_dir.glob("*.pth"):
            match = self.MODEL_PATTERN.match(model_path.name)
            if match:
                info = match.groupdict()
                site = info["site"]

                # Create unique ID
                model_id = f"dl_{site}"

                # Load metrics if available
                metrics_path = model_path.with_name(
                    model_path.stem + "_metrics.json"
                )
                metrics = None
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                self._model_registry[model_id] = {
                    "path": model_path,
                    "site": site,
                    "encoder": info["encoder"],
                    "architecture": info["arch"],
                    "input_config": info["input_config"],
                    "date": info["date"],
                    "metrics": metrics,
                }

    def load_all(self) -> None:
        """Load all discovered models into memory."""
        for model_id in self._model_registry:
            self.load_model(model_id)

    def load_model(self, model_id: str) -> DLModelHeuristic:
        """
        Load a specific model.

        Args:
            model_id: Model identifier (e.g., "dl_UCA")

        Returns:
            Loaded DLModelHeuristic instance
        """
        if model_id in self._heuristics:
            return self._heuristics[model_id]

        if model_id not in self._model_registry:
            raise KeyError(f"Unknown model: {model_id}. Available: {self.list_models()}")

        info = self._model_registry[model_id]

        heuristic = DLModelHeuristic(
            name=model_id,
            model_path=info["path"],
            device=self.device,
            threshold=self.threshold,
        )

        self._heuristics[model_id] = heuristic
        return heuristic

    def get_heuristic(self, model_id: str) -> DLModelHeuristic:
        """
        Get a loaded heuristic by ID.

        Args:
            model_id: Model identifier

        Returns:
            DLModelHeuristic instance
        """
        if model_id not in self._heuristics:
            return self.load_model(model_id)
        return self._heuristics[model_id]

    def get_all_heuristics(self) -> List[HeuristicBase]:
        """
        Get all loaded heuristics as a list (for LA framework).

        Returns:
            List of HeuristicBase instances
        """
        return list(self._heuristics.values())

    def list_models(self) -> List[str]:
        """
        List all available model IDs.

        Returns:
            List of model identifiers
        """
        return list(self._model_registry.keys())

    def get_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model metadata including metrics
        """
        if model_id not in self._model_registry:
            raise KeyError(f"Unknown model: {model_id}")
        return self._model_registry[model_id].copy()

    def get_site_model(self, site_code: str) -> Optional[DLModelHeuristic]:
        """
        Get the model trained for a specific site.

        Args:
            site_code: MGRS site code (e.g., "UCA", "UWT")

        Returns:
            DLModelHeuristic for the site, or None if not found
        """
        model_id = f"dl_{site_code}"
        if model_id in self._model_registry:
            return self.get_heuristic(model_id)
        return None

    def get_best_model_for_metric(
        self,
        metric: str = "test_kelp_dataset_iou",
        exclude_sites: Optional[List[str]] = None,
    ) -> Tuple[str, DLModelHeuristic]:
        """
        Get the model with best performance on a given metric.

        Args:
            metric: Metric name to optimize (default: test IoU)
            exclude_sites: Sites to exclude from selection

        Returns:
            Tuple of (model_id, heuristic)
        """
        exclude_sites = exclude_sites or []
        best_id = None
        best_value = -float("inf")

        for model_id, info in self._model_registry.items():
            site = info["site"]
            if site in exclude_sites:
                continue

            metrics = info.get("metrics")
            if metrics is None:
                continue

            # Check test_metrics first, then valid_metrics
            for metric_dict in metrics.get("test_metrics", []):
                if metric in metric_dict:
                    value = metric_dict[metric]
                    if value > best_value:
                        best_value = value
                        best_id = model_id

        if best_id is None:
            raise ValueError(f"No model found with metric: {metric}")

        return best_id, self.get_heuristic(best_id)

    def predict_with_all(
        self,
        tile: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run prediction with all loaded models.

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            Dictionary mapping model_id to prediction mask
        """
        results = {}
        for model_id, heuristic in self._heuristics.items():
            results[model_id] = heuristic.predict(tile)
        return results

    def predict_ensemble(
        self,
        tile: torch.Tensor,
        method: str = "mean",
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Generate ensemble prediction from all models.

        Args:
            tile: Input tensor of shape (C, H, W)
            method: Ensemble method ("mean", "vote", "weighted")
            weights: Model weights for weighted ensemble

        Returns:
            Binary mask of shape (H, W)
        """
        if not self._heuristics:
            raise ValueError("No models loaded")

        # Get probability maps from all models
        proba_maps = []
        weight_list = []

        for model_id, heuristic in self._heuristics.items():
            proba = heuristic.predict_proba(tile)
            proba_maps.append(proba)

            if weights is not None:
                weight_list.append(weights.get(model_id, 1.0))
            else:
                weight_list.append(1.0)

        proba_stack = torch.stack(proba_maps)
        weights_tensor = torch.tensor(weight_list).view(-1, 1, 1)

        if method == "mean":
            ensemble_proba = proba_stack.mean(dim=0)
        elif method == "vote":
            votes = (proba_stack > self.threshold).float()
            ensemble_proba = votes.mean(dim=0)
        elif method == "weighted":
            weighted = proba_stack * weights_tensor
            ensemble_proba = weighted.sum(dim=0) / weights_tensor.sum()
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return (ensemble_proba > self.threshold).float()

    def __len__(self) -> int:
        """Number of available models."""
        return len(self._model_registry)

    def __repr__(self) -> str:
        loaded = len(self._heuristics)
        total = len(self._model_registry)
        return f"HeuristicPool({loaded}/{total} models loaded)"
