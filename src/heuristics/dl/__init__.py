"""
Deep Learning Heuristics Module

Contains wrappers for deep learning models to be used as heuristics
in the Learning Automata framework.
"""

from src.heuristics.dl.heuristic_pool import (
    DLModelHeuristic,
    HeuristicPool,
)

__all__ = [
    "DLModelHeuristic",
    "HeuristicPool",
]
