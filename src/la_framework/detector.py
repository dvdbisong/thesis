"""
LA Detector Orchestrator

Combines the Learning Automata with a pool of heuristics to create
an adaptive kelp detection system.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from code.heuristics.base import HeuristicBase
from code.la_framework.automaton import LRIAutomaton, create_automaton, list_automaton_types
from code.la_framework.reward import RewardFunction


class LADetector:
    """
    Learning Automata-based Kelp Detector.

    Orchestrates the interaction between:
    - A pool of heuristics (spectral indices, ML models, etc.)
    - A learning automaton that selects which heuristic to use
    - A reward function that evaluates prediction quality

    The automaton learns to select the best heuristic over time
    based on feedback from the environment (ground truth labels).
    """

    def __init__(
        self,
        heuristics: List[HeuristicBase],
        alpha: float = 0.1,
        reward_type: str = "binary",
        iou_threshold: float = 0.3,
        seed: Optional[int] = None,
        automaton_type: str = "LR-I",
        automaton_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LA detector.

        Args:
            heuristics: List of heuristics to choose from
            alpha: Learning rate for the automaton
            reward_type: "binary" or "continuous"
            iou_threshold: Threshold for binary reward
            seed: Random seed for reproducibility
            automaton_type: Type of automaton ("LR-I", "LR-P", "VSLA", "Pursuit", etc.)
            automaton_config: Additional automaton-specific configuration
        """
        self.heuristics = heuristics
        self.heuristic_names = [h.name for h in heuristics]
        self.automaton_type = automaton_type

        # Build automaton config
        config = automaton_config or {}
        config["alpha"] = config.get("alpha", alpha)

        # Initialize automaton using factory
        self.automaton = create_automaton(
            automaton_type=automaton_type,
            n_actions=len(heuristics),
            config=config,
            seed=seed,
        )

        # Initialize reward function
        self.reward_fn = RewardFunction(
            reward_type=reward_type,
            iou_threshold=iou_threshold,
        )

        # Set up ensemble heuristic if present
        for h in self.heuristics:
            if h.name == "ensemble":
                # Give ensemble access to other heuristics
                other_heuristics = [hh for hh in self.heuristics if hh.name != "ensemble"]
                h.set_heuristics(other_heuristics)

        # Statistics
        self.step_count = 0

    def select_heuristic(self) -> Tuple[int, HeuristicBase]:
        """
        Select a heuristic using the automaton.

        Returns:
            Tuple of (action_index, heuristic)
        """
        action = self.automaton.select_action()
        return action, self.heuristics[action]

    def process_tile(self, tile: torch.Tensor) -> Tuple[torch.Tensor, int, str]:
        """
        Process a tile by selecting a heuristic and generating a prediction.

        Args:
            tile: Input tensor of shape (C, H, W)

        Returns:
            Tuple of (prediction, action_index, heuristic_name)
        """
        action, heuristic = self.select_heuristic()
        prediction = heuristic.predict(tile)
        return prediction, action, heuristic.name

    def receive_feedback(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        action: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Receive feedback from the environment and update the automaton.

        Args:
            prediction: The prediction that was made
            ground_truth: The ground truth mask
            action: The action (heuristic index) that was taken

        Returns:
            Tuple of (reward, metrics_dict)
        """
        # Compute reward
        reward, metrics = self.reward_fn.compute_reward(prediction, ground_truth)

        # Update automaton
        self.automaton.update(action, reward)

        self.step_count += 1

        return reward, metrics

    def process_and_learn(
        self, tile: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, int, float, Dict[str, float]]:
        """
        Process a tile and learn from the result.

        This is a convenience method that combines process_tile and receive_feedback.

        Args:
            tile: Input tensor of shape (C, H, W)
            ground_truth: Ground truth mask of shape (H, W)

        Returns:
            Tuple of (prediction, action, reward, metrics)
        """
        prediction, action, _ = self.process_tile(tile)
        reward, metrics = self.receive_feedback(prediction, ground_truth, action)
        return prediction, action, reward, metrics

    def evaluate_tile(
        self, tile: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """
        Evaluate a tile without updating the automaton.

        Uses the current best heuristic (highest probability) for evaluation.

        Args:
            tile: Input tensor of shape (C, H, W)
            ground_truth: Ground truth mask of shape (H, W)

        Returns:
            Tuple of (prediction, action, metrics)
        """
        # Use best action (greedy) instead of sampling
        action = self.automaton.get_best_action()
        heuristic = self.heuristics[action]
        prediction = heuristic.predict(tile)

        # Compute metrics without updating
        _, metrics = self.reward_fn.compute_reward(prediction, ground_truth)
        metrics["selected_heuristic"] = heuristic.name

        return prediction, action, metrics

    def get_probabilities(self) -> Dict[str, float]:
        """
        Get the current probability distribution over heuristics.

        Returns:
            Dictionary mapping heuristic name to probability
        """
        probs = self.automaton.get_probabilities()
        return {name: float(prob) for name, prob in zip(self.heuristic_names, probs)}

    def get_best_heuristic(self) -> Tuple[str, float]:
        """
        Get the current best heuristic and its probability.

        Returns:
            Tuple of (heuristic_name, probability)
        """
        action = self.automaton.get_best_action()
        prob = self.automaton.probs[action]
        return self.heuristic_names[action], float(prob)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the detector.

        Returns:
            Dictionary of statistics
        """
        automaton_stats = self.automaton.get_statistics()
        return {
            "step_count": self.step_count,
            "heuristic_names": self.heuristic_names,
            "probabilities": self.get_probabilities(),
            "best_heuristic": self.get_best_heuristic()[0],
            "automaton": automaton_stats,
        }

    def get_probability_history(self) -> List[Dict[str, float]]:
        """
        Get the history of probability distributions.

        Returns:
            List of probability dictionaries over time
        """
        history = []
        for probs in self.automaton.probability_history:
            history.append(
                {name: float(p) for name, p in zip(self.heuristic_names, probs)}
            )
        return history

    def reset(self, keep_probs: bool = False) -> None:
        """
        Reset the detector state.

        Args:
            keep_probs: If True, keep learned probabilities
        """
        self.automaton.reset(keep_probs=keep_probs)
        self.step_count = 0

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the detector.

        Returns:
            Configuration dictionary
        """
        automaton_config = {
            "type": self.automaton_type,
            "alpha": self.automaton.alpha,
            "n_actions": self.automaton.n_actions,
        }

        # Add algorithm-specific parameters
        if hasattr(self.automaton, "beta"):
            automaton_config["beta"] = self.automaton.beta
        if hasattr(self.automaton, "alpha_max"):
            automaton_config["alpha_max"] = self.automaton.alpha_max
            automaton_config["alpha_min"] = self.automaton.alpha_min
        if hasattr(self.automaton, "resolution"):
            automaton_config["resolution"] = self.automaton.resolution
        if hasattr(self.automaton, "prior_strength"):
            automaton_config["prior_strength"] = self.automaton.prior_strength

        return {
            "heuristics": [h.get_config() for h in self.heuristics],
            "automaton": automaton_config,
            "reward": self.reward_fn.get_config(),
        }

    @staticmethod
    def available_automaton_types() -> List[str]:
        """Return list of available automaton types."""
        return list_automaton_types()
