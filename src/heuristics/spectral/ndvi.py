"""
NDVI (Normalized Difference Vegetation Index) Heuristic

NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)

Higher values indicate more vegetation (including kelp).
"""

import torch

from src.heuristics.base import SpectralIndexHeuristic


class NDVIHeuristic(SpectralIndexHeuristic):
    """
    NDVI-based kelp detection heuristic.

    NDVI highlights vegetation by measuring the difference between
    near-infrared reflection (high for vegetation) and red absorption.
    """

    def __init__(
        self,
        threshold_method: str = "otsu",
        fixed_threshold: float = 0.3,
        **kwargs
    ):
        """
        Initialize NDVI heuristic.

        Args:
            threshold_method: "otsu" or "fixed"
            fixed_threshold: Threshold if using fixed method (typical: 0.2-0.4)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="ndvi",
            threshold_method=threshold_method,
            fixed_threshold=fixed_threshold,
            **kwargs
        )

    def compute_index(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Compute NDVI for the tile.

        NDVI = (B8 - B4) / (B8 + B4)

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            NDVI values of shape (H, W), range [-1, 1]
        """
        nir = self.get_band(tile, "B8")  # NIR (842nm)
        red = self.get_band(tile, "B4")  # Red (665nm)

        # Compute NDVI with epsilon to avoid division by zero
        eps = 1e-10
        ndvi = (nir - red) / (nir + red + eps)

        # Clip to valid range
        ndvi = torch.clamp(ndvi, -1.0, 1.0)

        return ndvi
