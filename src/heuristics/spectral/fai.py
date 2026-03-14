"""
FAI (Floating Algae Index) Heuristic

FAI = NIR - (Red + (SWIR - Red) * (λNIR - λRed) / (λSWIR - λRed))

For Sentinel-2:
FAI = B8 - (B4 + (B11 - B4) * (842 - 665) / (1610 - 665))
FAI = B8 - (B4 + (B11 - B4) * 0.1873)

FAI is specifically designed to detect floating vegetation/algae,
making it particularly suitable for floating kelp detection.
"""

import torch

from code.heuristics.base import SpectralIndexHeuristic


class FAIHeuristic(SpectralIndexHeuristic):
    """
    FAI-based kelp detection heuristic.

    FAI measures the deviation of NIR reflectance from a baseline
    formed by Red and SWIR bands. Positive values indicate floating
    vegetation like kelp.
    """

    # Wavelength ratio for Sentinel-2 bands
    # (λNIR - λRed) / (λSWIR - λRed) = (842 - 665) / (1610 - 665)
    WAVELENGTH_RATIO = (842 - 665) / (1610 - 665)  # ≈ 0.1873

    def __init__(
        self,
        threshold_method: str = "otsu",
        fixed_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize FAI heuristic.

        Args:
            threshold_method: "otsu" or "fixed"
            fixed_threshold: Threshold if using fixed method (typical: 0.0)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="fai",
            threshold_method=threshold_method,
            fixed_threshold=fixed_threshold,
            **kwargs
        )

    def compute_index(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Compute FAI for the tile.

        FAI = B8 - (B4 + (B11 - B4) * wavelength_ratio)

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            FAI values of shape (H, W)
        """
        nir = self.get_band(tile, "B8")    # NIR (842nm)
        red = self.get_band(tile, "B4")    # Red (665nm)
        swir = self.get_band(tile, "B11")  # SWIR (1610nm)

        # Compute baseline (linear interpolation between Red and SWIR)
        baseline = red + (swir - red) * self.WAVELENGTH_RATIO

        # FAI is deviation of NIR from baseline
        fai = nir - baseline

        return fai
