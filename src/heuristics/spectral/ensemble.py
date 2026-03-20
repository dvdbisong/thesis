"""
Ensemble Heuristic

Combines multiple spectral index heuristics using majority voting.
A pixel is classified as kelp only if the majority of heuristics agree.
"""

from typing import List, Optional

import torch

from src.heuristics.base import HeuristicBase


class EnsembleHeuristic(HeuristicBase):
    """
    Ensemble heuristic using majority voting.

    Combines predictions from multiple base heuristics and classifies
    a pixel as kelp only if a majority of heuristics predict kelp.
    """

    def __init__(
        self,
        heuristics: Optional[List[HeuristicBase]] = None,
        voting_method: str = "majority",
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize ensemble heuristic.

        Args:
            heuristics: List of base heuristics to combine.
                        If None, will be set later via set_heuristics()
            voting_method: "majority" for majority voting, "weighted" for future use
            threshold: Fraction of heuristics that must agree (default 0.5 = majority)
            **kwargs: Additional parameters
        """
        super().__init__(name="ensemble", **kwargs)
        self.heuristics = heuristics or []
        self.voting_method = voting_method
        self.threshold = threshold

    def set_heuristics(self, heuristics: List[HeuristicBase]):
        """
        Set the base heuristics for the ensemble.

        Args:
            heuristics: List of heuristics to combine
        """
        # Filter out self to avoid recursion
        self.heuristics = [h for h in heuristics if h.name != "ensemble"]

    def predict(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Generate ensemble prediction using majority voting.

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            Binary mask of shape (H, W) where 1=kelp, 0=no-kelp
        """
        if len(self.heuristics) == 0:
            raise RuntimeError("No heuristics set for ensemble. Call set_heuristics() first.")

        # Get predictions from all heuristics
        predictions = []
        for heuristic in self.heuristics:
            pred = heuristic.predict(tile)
            predictions.append(pred)

        # Stack predictions: (num_heuristics, H, W)
        stacked = torch.stack(predictions, dim=0)

        # Compute vote count per pixel
        vote_count = stacked.sum(dim=0)

        # Apply majority voting
        num_heuristics = len(self.heuristics)
        min_votes = int(num_heuristics * self.threshold)

        # Need at least min_votes to classify as kelp
        # For odd number of heuristics, this gives true majority
        mask = (vote_count > min_votes).float()

        return mask

    def get_config(self):
        config = super().get_config()
        config["voting_method"] = self.voting_method
        config["threshold"] = self.threshold
        config["base_heuristics"] = [h.name for h in self.heuristics]
        return config
