"""
Learning Automata Implementation

Implements various Learning Automata for adaptive heuristic selection:
- LR-I (Linear Reward-Inaction): Updates only on rewards
- LR-P (Linear Reward-Penalty): Updates on both rewards and penalties
- VSLA (Variable Structure LA): Adaptive learning rate based on entropy
- Pursuit: Pursues action with highest estimated reward
- Discretized Pursuit: Finite probability levels for faster convergence
- Estimator (SERI): Bayesian estimator with pursuit-style updates

The automaton maintains a probability vector over actions (heuristics)
and updates these probabilities based on feedback from the environment.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LRIAutomaton:
    """
    Linear Reward-Inaction (L_R-I) Learning Automaton.

    The L_R-I scheme:
    - On reward (positive feedback): increase probability of selected action
    - On penalty (negative feedback): no change (inaction)

    This makes the automaton robust to noisy environments and converges
    to the optimal action with probability 1 in stationary environments.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        initial_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the L_R-I automaton.

        Args:
            n_actions: Number of actions (heuristics) available
            alpha: Learning rate (reward parameter), typically 0.01-0.1
            initial_probs: Initial probability distribution. If None, uniform.
            seed: Random seed for reproducibility
        """
        self.n_actions = n_actions
        self.alpha = alpha

        # Initialize random state
        self.rng = np.random.RandomState(seed)

        # Initialize probability vector
        if initial_probs is not None:
            if len(initial_probs) != n_actions:
                raise ValueError(
                    f"initial_probs length ({len(initial_probs)}) must match "
                    f"n_actions ({n_actions})"
                )
            self.probs = np.array(initial_probs, dtype=np.float64)
            # Normalize to ensure valid probability distribution
            self.probs /= self.probs.sum()
        else:
            # Uniform distribution
            self.probs = np.ones(n_actions, dtype=np.float64) / n_actions

        # Statistics tracking
        self.action_counts = np.zeros(n_actions, dtype=np.int64)
        self.reward_counts = np.zeros(n_actions, dtype=np.int64)
        self.total_steps = 0
        self.probability_history: List[np.ndarray] = [self.probs.copy()]

    def select_action(self) -> int:
        """
        Select an action based on the current probability distribution.

        Returns:
            Index of the selected action
        """
        action = self.rng.choice(self.n_actions, p=self.probs)
        self.action_counts[action] += 1
        self.total_steps += 1
        return action

    def update(self, action: int, reward: float) -> None:
        """
        Update the probability vector based on feedback.

        L_R-I Update Rule:
        - If reward > 0 (success):
            p[action] = p[action] + alpha * (1 - p[action])
            p[other] = p[other] * (1 - alpha)  for all other actions
        - If reward <= 0 (failure):
            No change (inaction)

        Args:
            action: The action that was taken
            reward: The reward received (typically 0 or 1)
        """
        if reward > 0:
            # Reward: increase probability of selected action
            self.probs[action] = self.probs[action] + self.alpha * (1 - self.probs[action])

            # Decrease probability of other actions proportionally
            for i in range(self.n_actions):
                if i != action:
                    self.probs[i] = self.probs[i] * (1 - self.alpha)

            # Track rewards
            self.reward_counts[action] += 1

        # Inaction on penalty (reward <= 0): no update

        # Ensure probabilities sum to 1 (handle numerical errors)
        self.probs /= self.probs.sum()

        # Record probability history
        self.probability_history.append(self.probs.copy())

    def get_probabilities(self) -> np.ndarray:
        """
        Get the current probability distribution.

        Returns:
            Array of probabilities for each action
        """
        return self.probs.copy()

    def get_best_action(self) -> int:
        """
        Get the action with the highest probability.

        Returns:
            Index of the most probable action
        """
        return int(np.argmax(self.probs))

    def get_entropy(self) -> float:
        """
        Compute the entropy of the probability distribution.

        Lower entropy indicates more certainty (convergence).

        Returns:
            Shannon entropy in bits
        """
        # Avoid log(0)
        probs_safe = np.clip(self.probs, 1e-10, 1.0)
        entropy = -np.sum(probs_safe * np.log2(probs_safe))
        return entropy

    def get_statistics(self) -> dict:
        """
        Get statistics about the automaton's behavior.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_steps": self.total_steps,
            "action_counts": self.action_counts.tolist(),
            "reward_counts": self.reward_counts.tolist(),
            "current_probs": self.probs.tolist(),
            "best_action": self.get_best_action(),
            "entropy": self.get_entropy(),
            "reward_rate": (
                self.reward_counts / np.maximum(self.action_counts, 1)
            ).tolist(),
        }

    def reset(self, keep_probs: bool = False) -> None:
        """
        Reset the automaton state.

        Args:
            keep_probs: If True, keep current probabilities. If False, reset to uniform.
        """
        if not keep_probs:
            self.probs = np.ones(self.n_actions, dtype=np.float64) / self.n_actions

        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.reward_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.total_steps = 0
        self.probability_history = [self.probs.copy()]


class LRPAutomaton(LRIAutomaton):
    """
    Linear Reward-Penalty (L_R-P) Learning Automaton.

    Unlike L_R-I, this scheme also updates on penalties:
    - On reward: increase probability of selected action
    - On penalty: decrease probability of selected action

    This can lead to faster learning but may be less stable
    in noisy environments.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        beta: Optional[float] = None,
        initial_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the L_R-P automaton.

        Args:
            n_actions: Number of actions available
            alpha: Reward learning rate
            beta: Penalty learning rate. If None, uses alpha (symmetric).
            initial_probs: Initial probability distribution
            seed: Random seed
        """
        super().__init__(n_actions, alpha, initial_probs, seed)
        self.beta = beta if beta is not None else alpha

    def update(self, action: int, reward: float) -> None:
        """
        Update the probability vector based on feedback.

        L_R-P Update Rule:
        - If reward > 0 (success):
            p[action] = p[action] + alpha * (1 - p[action])
            p[other] = p[other] * (1 - alpha)
        - If reward <= 0 (failure):
            p[action] = p[action] * (1 - beta)
            p[other] = p[other] + beta / (n - 1) * (1 - p[other])

        Args:
            action: The action that was taken
            reward: The reward received
        """
        if reward > 0:
            # Same as L_R-I for reward
            self.probs[action] = self.probs[action] + self.alpha * (1 - self.probs[action])
            for i in range(self.n_actions):
                if i != action:
                    self.probs[i] = self.probs[i] * (1 - self.alpha)
            self.reward_counts[action] += 1
        else:
            # Penalty: decrease probability of selected action
            self.probs[action] = self.probs[action] * (1 - self.beta)

            # Increase probability of other actions
            beta_share = self.beta / (self.n_actions - 1)
            for i in range(self.n_actions):
                if i != action:
                    self.probs[i] = self.probs[i] + beta_share * (1 - self.probs[i])

        # Normalize
        self.probs /= self.probs.sum()
        self.probability_history.append(self.probs.copy())


class VSLAutomaton(LRIAutomaton):
    """
    Variable Structure Learning Automaton (VSLA).

    Adapts the learning rate based on the current entropy of the
    probability distribution:
    - High entropy (uncertain) → high alpha (explore more)
    - Low entropy (converged) → low alpha (exploit/stabilize)

    This allows fast initial learning while maintaining stability
    near convergence.
    """

    def __init__(
        self,
        n_actions: int,
        alpha_max: float = 0.1,
        alpha_min: float = 0.001,
        initial_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the VSLA.

        Args:
            n_actions: Number of actions available
            alpha_max: Maximum learning rate (high entropy)
            alpha_min: Minimum learning rate (low entropy)
            initial_probs: Initial probability distribution
            seed: Random seed
        """
        super().__init__(n_actions, alpha_max, initial_probs, seed)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.max_entropy = np.log2(n_actions)  # Entropy of uniform distribution

    def update(self, action: int, reward: float) -> None:
        """
        Update with adaptive learning rate based on entropy.

        Args:
            action: The action that was taken
            reward: The reward received
        """
        # Adapt alpha based on entropy
        current_entropy = self.get_entropy()
        entropy_ratio = current_entropy / self.max_entropy if self.max_entropy > 0 else 0
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * entropy_ratio

        # Then apply standard LR-I update
        super().update(action, reward)


class PursuitAutomaton(LRIAutomaton):
    """
    Pursuit Learning Automaton.

    Maintains maximum likelihood estimates of reward probability for each
    action and "pursues" (increases probability of) the action with the
    highest estimated reward.

    This leads to significantly faster convergence (10-20x) compared to
    standard LR-I, as it uses reward information more efficiently.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.01,
        initial_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Pursuit automaton.

        Args:
            n_actions: Number of actions available
            alpha: Learning rate for probability updates
            initial_probs: Initial probability distribution
            seed: Random seed
        """
        super().__init__(n_actions, alpha, initial_probs, seed)
        self.reward_estimates = np.zeros(n_actions, dtype=np.float64)
        self.action_samples = np.zeros(n_actions, dtype=np.int64)

    def update(self, action: int, reward: float) -> None:
        """
        Update reward estimates and pursue best action.

        Args:
            action: The action that was taken
            reward: The reward received
        """
        # Update ML estimate for selected action (running average)
        self.action_samples[action] += 1
        n = self.action_samples[action]
        self.reward_estimates[action] += (reward - self.reward_estimates[action]) / n

        # Track rewards
        if reward > 0:
            self.reward_counts[action] += 1

        # Pursue the action with highest estimated reward
        best_action = int(np.argmax(self.reward_estimates))

        # Increase probability of best estimated action
        self.probs[best_action] += self.alpha * (1 - self.probs[best_action])

        # Decrease probability of other actions
        for i in range(self.n_actions):
            if i != best_action:
                self.probs[i] *= (1 - self.alpha)

        # Normalize
        self.probs /= self.probs.sum()
        self.probability_history.append(self.probs.copy())

    def get_statistics(self) -> dict:
        """Get statistics including reward estimates."""
        stats = super().get_statistics()
        stats["reward_estimates"] = self.reward_estimates.tolist()
        stats["action_samples"] = self.action_samples.tolist()
        return stats

    def reset(self, keep_probs: bool = False) -> None:
        """Reset including estimates."""
        super().reset(keep_probs)
        self.reward_estimates = np.zeros(self.n_actions, dtype=np.float64)
        self.action_samples = np.zeros(self.n_actions, dtype=np.int64)


class DiscretizedPursuitAutomaton(PursuitAutomaton):
    """
    Discretized Pursuit Learning Automaton.

    Uses finite probability levels (resolution) instead of continuous
    probabilities. This can lead to even faster convergence than
    standard Pursuit.

    The probability space [0, 1] is divided into N equal levels,
    and updates move probability mass one level at a time.
    """

    def __init__(
        self,
        n_actions: int,
        resolution: int = 100,
        alpha: float = 0.01,
        initial_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Discretized Pursuit automaton.

        Args:
            n_actions: Number of actions available
            resolution: Number of probability levels (higher = finer)
            alpha: Learning rate (used for estimates, not probability updates)
            initial_probs: Initial probability distribution
            seed: Random seed
        """
        super().__init__(n_actions, alpha, initial_probs, seed)
        self.resolution = resolution
        # Initialize depth (discrete probability levels) uniformly
        self.depth = np.ones(n_actions, dtype=np.int64) * (resolution // n_actions)
        # Distribute remainder
        remainder = resolution - self.depth.sum()
        for i in range(int(remainder)):
            self.depth[i] += 1
        # Update probs to match depth
        self.probs = self.depth.astype(np.float64) / self.depth.sum()

    def update(self, action: int, reward: float) -> None:
        """
        Update with discretized probability movements.

        Args:
            action: The action that was taken
            reward: The reward received
        """
        # Update ML estimates (continuous)
        self.action_samples[action] += 1
        n = self.action_samples[action]
        self.reward_estimates[action] += (reward - self.reward_estimates[action]) / n

        if reward > 0:
            self.reward_counts[action] += 1

        # Find best estimated action
        best_action = int(np.argmax(self.reward_estimates))

        # Discretized update: move one level toward best action
        if self.depth[best_action] < self.resolution:
            # Find an action to take from (that has depth > 0 and is not best)
            others = [i for i in range(self.n_actions)
                      if i != best_action and self.depth[i] > 0]
            if others:
                # Take from random other action
                victim = self.rng.choice(others)
                self.depth[victim] -= 1
                self.depth[best_action] += 1

        # Update continuous probabilities from depth
        self.probs = self.depth.astype(np.float64) / self.depth.sum()
        self.probability_history.append(self.probs.copy())

    def get_statistics(self) -> dict:
        """Get statistics including depth levels."""
        stats = super().get_statistics()
        stats["depth"] = self.depth.tolist()
        stats["resolution"] = self.resolution
        return stats

    def reset(self, keep_probs: bool = False) -> None:
        """Reset including depth."""
        super().reset(keep_probs)
        self.depth = np.ones(self.n_actions, dtype=np.int64) * (self.resolution // self.n_actions)
        remainder = self.resolution - self.depth.sum()
        for i in range(int(remainder)):
            self.depth[i] += 1
        self.probs = self.depth.astype(np.float64) / self.depth.sum()


class EstimatorAutomaton(LRIAutomaton):
    """
    Stochastic Estimator Reward-Inaction (SERI) Learning Automaton.

    Uses Bayesian estimation to maintain posterior distributions over
    the reward probability for each action. Pursues the action with
    the highest posterior mean estimate.

    This is recognized as one of the fastest ε-optimal LA algorithms,
    combining the benefits of estimation with pursuit.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.01,
        prior_strength: float = 1.0,
        initial_probs: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Estimator automaton.

        Args:
            n_actions: Number of actions available
            alpha: Learning rate for probability updates
            prior_strength: Strength of Beta prior (higher = more conservative)
            initial_probs: Initial probability distribution
            seed: Random seed
        """
        super().__init__(n_actions, alpha, initial_probs, seed)
        # Beta distribution parameters (prior: Beta(1,1) = uniform)
        self.alpha_prior = np.ones(n_actions, dtype=np.float64) * prior_strength
        self.beta_prior = np.ones(n_actions, dtype=np.float64) * prior_strength

    def update(self, action: int, reward: float) -> None:
        """
        Update Bayesian estimates and pursue best action.

        Args:
            action: The action that was taken
            reward: The reward received
        """
        # Update Beta posterior for selected action
        # Treat reward > 0.5 as "success" for discrete interpretation
        if reward > 0.5:
            self.alpha_prior[action] += 1
            self.reward_counts[action] += 1
        else:
            self.beta_prior[action] += 1

        # Compute posterior mean estimates: E[θ] = α / (α + β)
        estimates = self.alpha_prior / (self.alpha_prior + self.beta_prior)

        # Pursue best estimated action
        best_action = int(np.argmax(estimates))

        # Increase probability of best
        self.probs[best_action] += self.alpha * (1 - self.probs[best_action])

        # Decrease probability of others
        for i in range(self.n_actions):
            if i != best_action:
                self.probs[i] *= (1 - self.alpha)

        # Normalize
        self.probs /= self.probs.sum()
        self.probability_history.append(self.probs.copy())

    def get_statistics(self) -> dict:
        """Get statistics including posterior parameters."""
        stats = super().get_statistics()
        estimates = self.alpha_prior / (self.alpha_prior + self.beta_prior)
        stats["posterior_alpha"] = self.alpha_prior.tolist()
        stats["posterior_beta"] = self.beta_prior.tolist()
        stats["posterior_mean"] = estimates.tolist()
        return stats

    def reset(self, keep_probs: bool = False) -> None:
        """Reset including posteriors."""
        super().reset(keep_probs)
        self.alpha_prior = np.ones(self.n_actions, dtype=np.float64)
        self.beta_prior = np.ones(self.n_actions, dtype=np.float64)


# =============================================================================
# Automaton Factory
# =============================================================================

AUTOMATON_REGISTRY = {
    "LR-I": LRIAutomaton,
    "LRI": LRIAutomaton,
    "LR-P": LRPAutomaton,
    "LRP": LRPAutomaton,
    "VSLA": VSLAutomaton,
    "Pursuit": PursuitAutomaton,
    "DiscretizedPursuit": DiscretizedPursuitAutomaton,
    "Estimator": EstimatorAutomaton,
    "SERI": EstimatorAutomaton,
}


def create_automaton(
    automaton_type: str,
    n_actions: int,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> LRIAutomaton:
    """
    Factory function to create an automaton of the specified type.

    Args:
        automaton_type: Type of automaton ("LR-I", "LR-P", "VSLA", "Pursuit", etc.)
        n_actions: Number of actions
        config: Configuration dictionary with algorithm-specific parameters
        seed: Random seed

    Returns:
        Automaton instance

    Raises:
        ValueError: If automaton_type is not recognized
    """
    config = config or {}

    if automaton_type not in AUTOMATON_REGISTRY:
        available = list(AUTOMATON_REGISTRY.keys())
        raise ValueError(
            f"Unknown automaton type: {automaton_type}. "
            f"Available: {available}"
        )

    automaton_class = AUTOMATON_REGISTRY[automaton_type]

    # Extract common parameters
    alpha = config.get("alpha", 0.01)
    initial_probs = config.get("initial_probs")

    # Create based on type
    if automaton_type in ["LR-I", "LRI"]:
        return LRIAutomaton(
            n_actions=n_actions,
            alpha=alpha,
            initial_probs=initial_probs,
            seed=seed,
        )

    elif automaton_type in ["LR-P", "LRP"]:
        beta = config.get("beta", alpha)
        return LRPAutomaton(
            n_actions=n_actions,
            alpha=alpha,
            beta=beta,
            initial_probs=initial_probs,
            seed=seed,
        )

    elif automaton_type == "VSLA":
        alpha_max = config.get("alpha_max", 0.1)
        alpha_min = config.get("alpha_min", 0.001)
        return VSLAutomaton(
            n_actions=n_actions,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            initial_probs=initial_probs,
            seed=seed,
        )

    elif automaton_type == "Pursuit":
        return PursuitAutomaton(
            n_actions=n_actions,
            alpha=alpha,
            initial_probs=initial_probs,
            seed=seed,
        )

    elif automaton_type == "DiscretizedPursuit":
        resolution = config.get("resolution", 100)
        return DiscretizedPursuitAutomaton(
            n_actions=n_actions,
            resolution=resolution,
            alpha=alpha,
            initial_probs=initial_probs,
            seed=seed,
        )

    elif automaton_type in ["Estimator", "SERI"]:
        prior_strength = config.get("prior_strength", 1.0)
        return EstimatorAutomaton(
            n_actions=n_actions,
            alpha=alpha,
            prior_strength=prior_strength,
            initial_probs=initial_probs,
            seed=seed,
        )

    else:
        # Fallback (should not reach here due to registry check)
        return automaton_class(
            n_actions=n_actions,
            alpha=alpha,
            initial_probs=initial_probs,
            seed=seed,
        )


def list_automaton_types() -> List[str]:
    """Return list of available automaton types."""
    return list(set(AUTOMATON_REGISTRY.keys()))
