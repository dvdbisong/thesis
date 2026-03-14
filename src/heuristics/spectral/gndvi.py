"""
GNDVI (Green Normalized Difference Vegetation Index) Heuristic

GNDVI = (NIR - Green) / (NIR + Green) = (B8 - B3) / (B8 + B3)

GNDVI uses the green band instead of red, making it more sensitive
to chlorophyll concentration. This can be beneficial for detecting
submerged or floating aquatic vegetation like kelp.
"""

import torch

from code.heuristics.base import SpectralIndexHeuristic


class GNDVIHeuristic(SpectralIndexHeuristic):
    """
    GNDVI-based kelp detection heuristic.

    GNDVI is more sensitive to chlorophyll content than NDVI,
    potentially better for detecting kelp in varying water conditions.
    """

    def __init__(
        self,
        threshold_method: str = "otsu",
        fixed_threshold: float = 0.3,
        **kwargs
    ):
        """
        Initialize GNDVI heuristic.

        Args:
            threshold_method: "otsu" or "fixed"
            fixed_threshold: Threshold if using fixed method (typical: 0.2-0.4)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="gndvi",
            threshold_method=threshold_method,
            fixed_threshold=fixed_threshold,
            **kwargs
        )

    def compute_index(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Compute GNDVI for the tile.

        GNDVI = (B8 - B3) / (B8 + B3)

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            GNDVI values of shape (H, W), range [-1, 1]
        """
        nir = self.get_band(tile, "B8")    # NIR (842nm)
        green = self.get_band(tile, "B3")  # Green (560nm)

        # Compute GNDVI with epsilon to avoid division by zero
        eps = 1e-10
        gndvi = (nir - green) / (nir + green + eps)

        # Clip to valid range
        gndvi = torch.clamp(gndvi, -1.0, 1.0)

        return gndvi
