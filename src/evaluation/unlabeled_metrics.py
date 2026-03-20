"""
Proxy Evaluation Metrics for Unlabeled Multi-Temporal Data.

Since multi-temporal images have no ground truth labels, we use proxy metrics
to evaluate the LA framework's performance:

1. Temporal Coherence: Prediction consistency within short time windows
2. Phenological Plausibility: Matches expected seasonal kelp patterns
3. Heuristic Agreement: Cross-model consistency on the same tile
4. LA Convergence: Policy stability and entropy over time

These metrics enable evaluation of the dual-layer MSE/PSE framework
without requiring dense ground truth labels.

Author: Ekaba Bisong
Date: March 2026
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TemporalCoherenceResult:
    """Result of temporal coherence analysis."""
    mean_coherence: float
    std_coherence: float
    n_pairs: int
    coherence_by_window: Dict[int, float]


@dataclass
class PhenologicalResult:
    """Result of phenological plausibility analysis."""
    plausibility_score: float
    seasonal_pattern: Dict[str, float]
    matches_expected: bool
    summer_winter_ratio: float


@dataclass
class HeuristicAgreementResult:
    """Result of heuristic agreement analysis."""
    mean_agreement: float
    pairwise_agreement: Dict[Tuple[str, str], float]
    most_agreeing_pair: Tuple[str, str]
    least_agreeing_pair: Tuple[str, str]


@dataclass
class LAConvergenceResult:
    """Result of LA convergence analysis."""
    converged: bool
    steps_to_convergence: int
    final_entropy: float
    entropy_history: List[float]
    dominant_heuristic: str
    probability_distribution: Dict[str, float]


def compute_iou(pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    Args:
        pred1: First binary prediction mask
        pred2: Second binary prediction mask

    Returns:
        IoU score in [0, 1]
    """
    pred1 = pred1.astype(bool)
    pred2 = pred2.astype(bool)

    intersection = np.logical_and(pred1, pred2).sum()
    union = np.logical_or(pred1, pred2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_dice(pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred1: First binary prediction mask
        pred2: Second binary prediction mask

    Returns:
        Dice score in [0, 1]
    """
    pred1 = pred1.astype(bool)
    pred2 = pred2.astype(bool)

    intersection = np.logical_and(pred1, pred2).sum()
    total = pred1.sum() + pred2.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2 * intersection / total


def temporal_coherence(
    predictions: List[np.ndarray],
    timestamps: List[datetime],
    window_days: int = 30,
    metric: str = "iou",
) -> TemporalCoherenceResult:
    """
    Measure prediction consistency within time windows.

    High-quality LA should produce similar predictions for images close
    in time, as kelp canopy doesn't change rapidly (days to weeks).

    Args:
        predictions: List of binary prediction masks
        timestamps: List of datetime objects for each prediction
        window_days: Maximum days between images to compare
        metric: Similarity metric ("iou" or "dice")

    Returns:
        TemporalCoherenceResult with coherence statistics
    """
    if len(predictions) != len(timestamps):
        raise ValueError("Number of predictions must match number of timestamps")

    similarity_fn = compute_iou if metric == "iou" else compute_dice
    coherence_scores = []
    coherence_by_window = {}

    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            delta_days = abs((timestamps[j] - timestamps[i]).days)

            if delta_days <= window_days:
                score = similarity_fn(predictions[i], predictions[j])
                coherence_scores.append(score)

                # Track by window size
                window_bucket = (delta_days // 7) * 7  # Weekly buckets
                if window_bucket not in coherence_by_window:
                    coherence_by_window[window_bucket] = []
                coherence_by_window[window_bucket].append(score)

    # Average by window bucket
    coherence_by_window_avg = {
        k: np.mean(v) for k, v in coherence_by_window.items()
    }

    return TemporalCoherenceResult(
        mean_coherence=np.mean(coherence_scores) if coherence_scores else 0.0,
        std_coherence=np.std(coherence_scores) if coherence_scores else 0.0,
        n_pairs=len(coherence_scores),
        coherence_by_window=coherence_by_window_avg,
    )


def phenological_plausibility(
    monthly_extents: Dict[int, float],
    expected_pattern: str = "temperate",
) -> PhenologicalResult:
    """
    Check if predicted kelp extent matches expected seasonal pattern.

    Temperate kelp forests (like BC) exhibit:
    - Peak canopy extent in summer/fall (Jun-Sep)
    - Minimum extent in winter (Dec-Feb)
    - Rapid growth in spring (Mar-May)

    Args:
        monthly_extents: Dict mapping month (1-12) to mean kelp extent (area or %)
        expected_pattern: Expected pattern type ("temperate", "tropical", "polar")

    Returns:
        PhenologicalResult with plausibility assessment
    """
    # Define expected patterns
    patterns = {
        "temperate": {
            "peak_months": [6, 7, 8, 9],      # Summer/early fall
            "trough_months": [12, 1, 2],       # Winter
            "growth_months": [3, 4, 5],        # Spring
            "decline_months": [10, 11],        # Late fall
        },
        "tropical": {
            "peak_months": [1, 2, 3],          # Cooler season
            "trough_months": [7, 8, 9],        # Warmer season
        },
        "polar": {
            "peak_months": [7, 8, 9],          # Brief summer
            "trough_months": [1, 2, 3, 10, 11, 12],  # Extended winter
        },
    }

    if expected_pattern not in patterns:
        expected_pattern = "temperate"

    pattern = patterns[expected_pattern]

    # Calculate seasonal means
    def get_seasonal_mean(months: List[int]) -> float:
        values = [monthly_extents.get(m, 0) for m in months]
        return np.mean(values) if values else 0

    seasonal_pattern = {}
    if expected_pattern == "temperate":
        seasonal_pattern = {
            "winter": get_seasonal_mean([12, 1, 2]),
            "spring": get_seasonal_mean([3, 4, 5]),
            "summer": get_seasonal_mean([6, 7, 8]),
            "fall": get_seasonal_mean([9, 10, 11]),
        }

    # Calculate peak and trough
    peak_extent = get_seasonal_mean(pattern["peak_months"])
    trough_extent = get_seasonal_mean(pattern["trough_months"])

    # Check if pattern matches expectations
    if expected_pattern == "temperate":
        # Summer should exceed winter
        matches_expected = peak_extent > trough_extent
    else:
        matches_expected = peak_extent >= trough_extent

    # Calculate ratio
    summer_winter_ratio = (
        peak_extent / trough_extent if trough_extent > 0 else float('inf')
    )

    # Plausibility score (0-1)
    # Higher score if pattern matches expected seasonal dynamics
    if matches_expected:
        # Normalize ratio to score (cap at 3x for max score)
        ratio_score = min(1.0, (summer_winter_ratio - 1) / 2)
        plausibility_score = 0.5 + 0.5 * ratio_score
    else:
        # Inverted pattern - low score
        plausibility_score = max(0.0, 0.5 - (trough_extent / peak_extent - 1) / 2) if peak_extent > 0 else 0.0

    return PhenologicalResult(
        plausibility_score=plausibility_score,
        seasonal_pattern=seasonal_pattern,
        matches_expected=matches_expected,
        summer_winter_ratio=summer_winter_ratio,
    )


def heuristic_agreement(
    predictions: Dict[str, np.ndarray],
    metric: str = "iou",
) -> HeuristicAgreementResult:
    """
    Measure agreement between different heuristics on the same input.

    High agreement suggests the detection task is "easy" for that input.
    Low agreement suggests challenging conditions where LA's adaptive
    selection provides most value.

    Args:
        predictions: Dict mapping heuristic name to binary prediction
        metric: Similarity metric ("iou" or "dice")

    Returns:
        HeuristicAgreementResult with agreement statistics
    """
    similarity_fn = compute_iou if metric == "iou" else compute_dice
    heuristic_names = list(predictions.keys())

    if len(heuristic_names) < 2:
        return HeuristicAgreementResult(
            mean_agreement=1.0,
            pairwise_agreement={},
            most_agreeing_pair=(heuristic_names[0], heuristic_names[0]) if heuristic_names else ("", ""),
            least_agreeing_pair=(heuristic_names[0], heuristic_names[0]) if heuristic_names else ("", ""),
        )

    pairwise_agreement = {}
    all_scores = []

    for i, h1 in enumerate(heuristic_names):
        for j, h2 in enumerate(heuristic_names):
            if i < j:
                score = similarity_fn(predictions[h1], predictions[h2])
                pairwise_agreement[(h1, h2)] = score
                all_scores.append((score, h1, h2))

    # Find extremes
    all_scores.sort()
    least_agreeing = (all_scores[0][1], all_scores[0][2])
    most_agreeing = (all_scores[-1][1], all_scores[-1][2])

    return HeuristicAgreementResult(
        mean_agreement=np.mean([s[0] for s in all_scores]),
        pairwise_agreement=pairwise_agreement,
        most_agreeing_pair=most_agreeing,
        least_agreeing_pair=least_agreeing,
    )


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    Args:
        probabilities: Array of probabilities (should sum to 1)

    Returns:
        Entropy value (0 = deterministic, log(n) = uniform)
    """
    # Avoid log(0)
    p = probabilities[probabilities > 0]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log(p))


def la_convergence_metrics(
    probability_history: List[np.ndarray],
    heuristic_names: List[str],
    convergence_threshold: float = 0.05,
    window_size: int = 50,
) -> LAConvergenceResult:
    """
    Analyze LA convergence from probability history.

    Convergence is detected when the probability distribution
    stops changing significantly (low variance over recent history).

    Args:
        probability_history: List of probability vectors over time
        heuristic_names: Names of heuristics
        convergence_threshold: Max std for convergence
        window_size: Window size for convergence detection

    Returns:
        LAConvergenceResult with convergence analysis
    """
    n_steps = len(probability_history)
    n_heuristics = len(heuristic_names)

    if n_steps == 0:
        return LAConvergenceResult(
            converged=False,
            steps_to_convergence=-1,
            final_entropy=0.0,
            entropy_history=[],
            dominant_heuristic="",
            probability_distribution={},
        )

    # Compute entropy history
    entropy_history = [compute_entropy(p) for p in probability_history]

    # Detect convergence point
    converged = False
    steps_to_convergence = n_steps

    for i in range(window_size, n_steps):
        window = probability_history[i - window_size:i]
        window_array = np.array(window)

        # Check if all probabilities are stable
        std_per_heuristic = np.std(window_array, axis=0)
        if np.all(std_per_heuristic < convergence_threshold):
            converged = True
            steps_to_convergence = i
            break

    # Final distribution
    final_probs = probability_history[-1]
    final_entropy = entropy_history[-1]

    # Dominant heuristic
    dominant_idx = np.argmax(final_probs)
    dominant_heuristic = heuristic_names[dominant_idx]

    # Create distribution dict
    probability_distribution = {
        heuristic_names[i]: float(final_probs[i])
        for i in range(n_heuristics)
    }

    return LAConvergenceResult(
        converged=converged,
        steps_to_convergence=steps_to_convergence,
        final_entropy=final_entropy,
        entropy_history=entropy_history,
        dominant_heuristic=dominant_heuristic,
        probability_distribution=probability_distribution,
    )


def compute_all_metrics(
    predictions_by_time: Dict[datetime, Dict[str, np.ndarray]],
    la_probability_history: List[np.ndarray],
    heuristic_names: List[str],
    selected_heuristic_per_time: Dict[datetime, str],
) -> Dict:
    """
    Compute all proxy metrics for a multi-temporal evaluation.

    Args:
        predictions_by_time: Nested dict of predictions keyed by timestamp and heuristic
        la_probability_history: LA probability vectors over time
        heuristic_names: Names of heuristics
        selected_heuristic_per_time: Which heuristic LA selected at each time

    Returns:
        Dictionary with all computed metrics
    """
    timestamps = sorted(predictions_by_time.keys())

    # 1. Temporal Coherence (using LA's selected predictions)
    selected_predictions = [
        predictions_by_time[t][selected_heuristic_per_time[t]]
        for t in timestamps
        if t in selected_heuristic_per_time
    ]
    selected_timestamps = [
        t for t in timestamps if t in selected_heuristic_per_time
    ]

    coherence = temporal_coherence(
        selected_predictions,
        selected_timestamps,
        window_days=30,
    )

    # 2. Phenological Plausibility
    monthly_extents = {}
    for t in timestamps:
        month = t.month
        heuristic = selected_heuristic_per_time.get(t, heuristic_names[0])
        pred = predictions_by_time[t][heuristic]
        extent = np.mean(pred)  # Fraction of kelp pixels

        if month not in monthly_extents:
            monthly_extents[month] = []
        monthly_extents[month].append(extent)

    monthly_extents_mean = {m: np.mean(v) for m, v in monthly_extents.items()}
    phenology = phenological_plausibility(monthly_extents_mean)

    # 3. Heuristic Agreement (average across all times)
    all_agreements = []
    for t in timestamps:
        agreement = heuristic_agreement(predictions_by_time[t])
        all_agreements.append(agreement.mean_agreement)

    mean_agreement = np.mean(all_agreements)

    # 4. LA Convergence
    convergence = la_convergence_metrics(
        la_probability_history,
        heuristic_names,
    )

    return {
        "temporal_coherence": {
            "mean": coherence.mean_coherence,
            "std": coherence.std_coherence,
            "n_pairs": coherence.n_pairs,
            "by_window": coherence.coherence_by_window,
        },
        "phenological_plausibility": {
            "score": phenology.plausibility_score,
            "seasonal_pattern": phenology.seasonal_pattern,
            "matches_expected": phenology.matches_expected,
            "summer_winter_ratio": phenology.summer_winter_ratio,
        },
        "heuristic_agreement": {
            "mean": mean_agreement,
        },
        "la_convergence": {
            "converged": convergence.converged,
            "steps": convergence.steps_to_convergence,
            "final_entropy": convergence.final_entropy,
            "dominant_heuristic": convergence.dominant_heuristic,
            "distribution": convergence.probability_distribution,
        },
    }
