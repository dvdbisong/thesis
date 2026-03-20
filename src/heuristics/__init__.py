"""
Heuristics Module

Provides heuristics for kelp detection:
- Spectral index heuristics (NDVI, FAI, GNDVI, etc.)
- Deep learning model heuristics (UNet-MaxViT)
- Ensemble heuristics
"""

from src.heuristics.base import HeuristicBase, SpectralIndexHeuristic

__all__ = [
    "HeuristicBase",
    "SpectralIndexHeuristic",
]
