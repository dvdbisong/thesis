"""
Heuristic Base Interface

Defines the common interface that all heuristics (spectral indices, ML models, etc.)
must implement for use with the Learning Automata framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class HeuristicBase(ABC):
    """
    Abstract base class for all heuristics.

    A heuristic takes a multi-spectral tile and produces a binary kelp/no-kelp mask.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the heuristic.

        Args:
            name: Unique identifier for this heuristic
            **kwargs: Additional heuristic-specific parameters
        """
        self.name = name
        self.params = kwargs

    @abstractmethod
    def predict(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Generate a binary prediction mask for the input tile.

        Args:
            tile: Input tensor of shape (C, H, W) with spectral bands

        Returns:
            Binary mask of shape (H, W) where 1=kelp, 0=no-kelp
        """
        pass

    def requires_training(self) -> bool:
        """
        Whether this heuristic requires training before use.

        Returns:
            True if the heuristic needs to be trained, False otherwise
        """
        return False

    def train(self, train_data: Any) -> None:
        """
        Train the heuristic (if required).

        Args:
            train_data: Training data (format depends on heuristic type)
        """
        if self.requires_training():
            raise NotImplementedError(
                f"Heuristic {self.name} requires training but train() is not implemented"
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this heuristic.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "requires_training": self.requires_training(),
            "params": self.params,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class SpectralIndexHeuristic(HeuristicBase):
    """
    Base class for spectral index-based heuristics.

    Spectral indices compute a ratio or combination of spectral bands
    and threshold the result to produce a binary mask.
    """

    # Band name to index mapping (for normalized 10-band input)
    BAND_INDICES = {
        "B2": 0,   # Blue
        "B3": 1,   # Green
        "B4": 2,   # Red
        "B5": 3,   # Red Edge 1
        "B6": 4,   # Red Edge 2
        "B7": 5,   # Red Edge 3
        "B8": 6,   # NIR
        "B8A": 7,  # Red Edge 4
        "B11": 8,  # SWIR 1
        "B12": 9,  # SWIR 2
    }

    def __init__(
        self,
        name: str,
        threshold_method: str = "otsu",
        fixed_threshold: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize spectral index heuristic.

        Args:
            name: Heuristic name
            threshold_method: "otsu" for automatic or "fixed" for manual threshold
            fixed_threshold: Threshold value if method is "fixed"
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.threshold_method = threshold_method
        self.fixed_threshold = fixed_threshold

    def get_band(self, tile: torch.Tensor, band_name: str) -> torch.Tensor:
        """
        Extract a specific band from the tile.

        Args:
            tile: Input tensor of shape (C, H, W)
            band_name: Band name (e.g., "B8", "B4")

        Returns:
            Band tensor of shape (H, W)
        """
        if band_name not in self.BAND_INDICES:
            raise ValueError(f"Unknown band: {band_name}")
        idx = self.BAND_INDICES[band_name]
        return tile[idx]

    @abstractmethod
    def compute_index(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral index for the tile.

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            Index values of shape (H, W)
        """
        pass

    def compute_threshold(self, index: torch.Tensor) -> float:
        """
        Compute the threshold for binarization.

        Args:
            index: Spectral index values of shape (H, W)

        Returns:
            Threshold value
        """
        if self.threshold_method == "fixed":
            if self.fixed_threshold is None:
                raise ValueError("fixed_threshold must be set when using 'fixed' method")
            return self.fixed_threshold

        elif self.threshold_method == "otsu":
            return self._otsu_threshold(index)

        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    def _otsu_threshold(self, index: torch.Tensor) -> float:
        """
        Compute Otsu's threshold.

        Args:
            index: Spectral index values

        Returns:
            Optimal threshold
        """
        # Flatten and convert to numpy for histogram computation
        values = index.flatten().numpy()

        # Handle NaN/Inf values
        values = values[~(torch.isnan(torch.tensor(values)) | torch.isinf(torch.tensor(values))).numpy()]

        if len(values) == 0:
            return 0.0

        # Compute histogram
        hist, bin_edges = torch.histogram(torch.tensor(values), bins=256)
        hist = hist.float()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize histogram
        hist = hist / hist.sum()

        # Compute cumulative sums and means
        cum_sum = torch.cumsum(hist, dim=0)
        cum_mean = torch.cumsum(hist * bin_centers, dim=0)
        global_mean = cum_mean[-1]

        # Compute between-class variance
        weight_bg = cum_sum
        weight_fg = 1 - cum_sum

        # Avoid division by zero
        weight_bg = torch.clamp(weight_bg, min=1e-10)
        weight_fg = torch.clamp(weight_fg, min=1e-10)

        mean_bg = cum_mean / weight_bg
        mean_fg = (global_mean - cum_mean) / weight_fg

        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        # Find threshold that maximizes between-class variance
        max_idx = torch.argmax(between_variance)
        threshold = bin_centers[max_idx].item()

        return threshold

    def predict(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Generate binary mask from spectral index.

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            Binary mask of shape (H, W)
        """
        # Compute spectral index
        index = self.compute_index(tile)

        # Compute threshold
        threshold = self.compute_threshold(index)

        # Binarize
        mask = (index > threshold).float()

        return mask

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["threshold_method"] = self.threshold_method
        config["fixed_threshold"] = self.fixed_threshold
        return config
