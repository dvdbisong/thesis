"""
Main Experiment Runner

Runs the Learning Automata kelp detection experiment with configurable parameters.
Handles training, evaluation, logging, and visualization.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from code.experiments.tracker import ExperimentTracker, load_config
from code.heuristics.spectral import (
    EnsembleHeuristic,
    FAIHeuristic,
    GNDVIHeuristic,
    NDVIHeuristic,
)
from code.la_framework.detector import LADetector
from code.la_framework.reward import compute_metrics
from code.preprocessing.data_loader import (
    KelpTileDataset,
    create_splits,
    load_splits,
)


def load_baseline_results(results_path: str = "results/baselines.json") -> Optional[Dict[str, Any]]:
    """
    Load baseline results if available.

    Args:
        results_path: Path to baselines.json

    Returns:
        Baseline results dictionary or None if not found
    """
    path = Path(results_path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def print_baseline_comparison(la_test_iou: float, baselines: Dict[str, Any]) -> None:
    """
    Print comparison table between LA and baselines.

    Args:
        la_test_iou: LA test IoU score
        baselines: Baseline results dictionary
    """
    print("\n=== Comparison with Baselines ===")
    print(f"{'Method':<20} {'IoU':<10} {'vs LA':<15}")
    print("-" * 45)

    # Collect all methods with their IoU
    methods = []

    # Add baselines
    for name, result in baselines.items():
        iou = result.get("iou", 0)
        methods.append((name, iou))

    # Add LA
    methods.append(("LA (learned)", la_test_iou))

    # Sort by IoU descending
    methods.sort(key=lambda x: x[1], reverse=True)

    # Print table
    for name, iou in methods:
        if name == "LA (learned)":
            diff = ""
            print(f"{name:<20} {iou:<10.4f} {diff:<15} <-- THIS RUN")
        else:
            diff = la_test_iou - iou
            if diff > 0:
                diff_str = f"+{diff:.4f}"
            else:
                diff_str = f"{diff:.4f}"
            print(f"{name:<20} {iou:<10.4f} {diff_str:<15}")

    # Summary
    random_iou = baselines.get("random", {}).get("iou", 0)
    oracle_iou = baselines.get("oracle", {}).get("iou", 0)

    print("-" * 45)
    if la_test_iou > random_iou:
        print(f"✓ LA outperforms random baseline by {la_test_iou - random_iou:.4f}")
    else:
        print(f"✗ LA underperforms random baseline by {random_iou - la_test_iou:.4f}")

    if oracle_iou > 0:
        gap_to_oracle = oracle_iou - la_test_iou
        print(f"  Gap to oracle: {gap_to_oracle:.4f} ({gap_to_oracle/oracle_iou*100:.1f}% of oracle IoU)")


def create_heuristics(config: Dict[str, Any]) -> List:
    """
    Create heuristics based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of heuristic instances
    """
    heuristic_config = config.get("heuristics", {})
    enabled = heuristic_config.get("enabled", ["ndvi", "fai", "gndvi", "ensemble"])

    heuristics = []

    if "ndvi" in enabled:
        ndvi_cfg = heuristic_config.get("ndvi", {})
        heuristics.append(
            NDVIHeuristic(
                threshold_method=ndvi_cfg.get("threshold_method", "otsu"),
                fixed_threshold=ndvi_cfg.get("fixed_threshold", 0.3),
            )
        )

    if "fai" in enabled:
        fai_cfg = heuristic_config.get("fai", {})
        heuristics.append(
            FAIHeuristic(
                threshold_method=fai_cfg.get("threshold_method", "otsu"),
                fixed_threshold=fai_cfg.get("fixed_threshold", 0.0),
            )
        )

    if "gndvi" in enabled:
        gndvi_cfg = heuristic_config.get("gndvi", {})
        heuristics.append(
            GNDVIHeuristic(
                threshold_method=gndvi_cfg.get("threshold_method", "otsu"),
                fixed_threshold=gndvi_cfg.get("fixed_threshold", 0.3),
            )
        )

    if "ensemble" in enabled:
        heuristics.append(EnsembleHeuristic())

    return heuristics


def run_training(
    detector: LADetector,
    train_data: List[Dict],
    data_dir: str,
    config: Dict[str, Any],
    tracker: ExperimentTracker,
) -> Dict[str, Any]:
    """
    Run the training loop.

    Args:
        detector: LA detector instance
        train_data: List of training tile paths
        data_dir: Path to data directory
        config: Configuration dictionary
        tracker: Experiment tracker

    Returns:
        Training statistics
    """
    training_cfg = config.get("training", {})
    num_epochs = training_cfg.get("num_epochs", 10)
    log_interval = training_cfg.get("log_interval", 50)
    shuffle = training_cfg.get("shuffle", True)

    # Create dataset
    dataset = KelpTileDataset(data_dir, tile_paths=train_data)

    # Training statistics
    epoch_stats = []
    all_rewards = []
    all_ious = []

    for epoch in range(num_epochs):
        # Shuffle data if requested
        indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(indices)

        epoch_rewards = []
        epoch_ious = []

        pbar = tqdm(indices, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, idx in enumerate(pbar):
            tile, mask, metadata = dataset[idx]

            # Process tile and learn
            pred, action, reward, metrics = detector.process_and_learn(tile, mask)

            epoch_rewards.append(reward)
            epoch_ious.append(metrics["iou"])
            all_rewards.append(reward)
            all_ious.append(metrics["iou"])

            # Log periodically
            if (step + 1) % log_interval == 0:
                probs = detector.get_probabilities()
                tracker.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "reward": reward,
                        "iou": metrics["iou"],
                        "probabilities": probs,
                    },
                    step=detector.step_count,
                )

            # Update progress bar
            pbar.set_postfix(
                {
                    "reward": f"{np.mean(epoch_rewards[-100:]):.2f}",
                    "iou": f"{np.mean(epoch_ious[-100:]):.4f}",
                }
            )

        # Epoch statistics
        stats = {
            "epoch": epoch,
            "mean_reward": np.mean(epoch_rewards),
            "mean_iou": np.mean(epoch_ious),
            "probabilities": detector.get_probabilities(),
            "entropy": detector.automaton.get_entropy(),
        }
        epoch_stats.append(stats)

        print(
            f"Epoch {epoch + 1}: "
            f"reward={stats['mean_reward']:.3f}, "
            f"iou={stats['mean_iou']:.4f}, "
            f"entropy={stats['entropy']:.4f}"
        )
        print(f"  Probabilities: {stats['probabilities']}")

    return {
        "epoch_stats": epoch_stats,
        "final_probabilities": detector.get_probabilities(),
        "total_rewards": sum(all_rewards),
        "mean_reward": np.mean(all_rewards),
        "mean_iou": np.mean(all_ious),
    }


def run_evaluation(
    detector: LADetector,
    test_data: List[Dict],
    data_dir: str,
    name: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate the detector on a test set.

    Args:
        detector: LA detector instance
        test_data: List of test tile paths
        data_dir: Path to data directory
        name: Name for this evaluation (e.g., "val", "test")

    Returns:
        Evaluation metrics
    """
    dataset = KelpTileDataset(data_dir, tile_paths=test_data)

    all_metrics = []
    heuristic_counts = {}

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {name}"):
        tile, mask, metadata = dataset[idx]

        # Evaluate without learning
        pred, action, metrics = detector.evaluate_tile(tile, mask)
        all_metrics.append(metrics)

        heuristic_name = detector.heuristic_names[action]
        heuristic_counts[heuristic_name] = heuristic_counts.get(heuristic_name, 0) + 1

    # Aggregate metrics
    iou_scores = [m["iou"] for m in all_metrics]
    f1_scores = [m["f1"] for m in all_metrics]
    precision_scores = [m["precision"] for m in all_metrics]
    recall_scores = [m["recall"] for m in all_metrics]

    return {
        f"{name}_iou": np.mean(iou_scores),
        f"{name}_iou_std": np.std(iou_scores),
        f"{name}_f1": np.mean(f1_scores),
        f"{name}_precision": np.mean(precision_scores),
        f"{name}_recall": np.mean(recall_scores),
        f"{name}_heuristic_usage": heuristic_counts,
        f"{name}_num_samples": len(test_data),
    }


def plot_convergence(
    detector: LADetector, tracker: ExperimentTracker, config: Dict[str, Any]
):
    """
    Plot probability convergence over time.

    Args:
        detector: LA detector instance
        tracker: Experiment tracker
        config: Configuration dictionary
    """
    history = detector.automaton.probability_history
    heuristic_names = detector.heuristic_names

    # Convert to arrays
    probs = np.array(history)  # (num_steps, num_heuristics)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot probability evolution
    for i, name in enumerate(heuristic_names):
        ax1.plot(probs[:, i], label=name, linewidth=1.5)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Probability")
    ax1.set_title("Heuristic Selection Probability Over Time")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot entropy
    entropies = []
    for p in probs:
        p_safe = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p_safe * np.log2(p_safe))
        entropies.append(entropy)

    ax2.plot(entropies, color="purple", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Entropy (bits)")
    ax2.set_title("Probability Distribution Entropy")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.log2(len(heuristic_names)), color="r", linestyle="--",
                label=f"Max entropy ({np.log2(len(heuristic_names)):.2f})")
    ax2.legend()

    plt.tight_layout()

    # Save plot
    output_cfg = config.get("output", {})
    if output_cfg.get("save_plots", True):
        format = output_cfg.get("plot_format", "png")
        dpi = output_cfg.get("plot_dpi", 150)
        tracker.save_plot(fig, "convergence", format=format, dpi=dpi)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run LA kelp detection experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom experiment name",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Set random seed
    seed = config.get("experiment", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiments_dir="experiments",
        experiment_name=args.name,
    )
    exp_dir = tracker.start_experiment(config)

    try:
        # Get data configuration
        data_cfg = config.get("data", {})
        data_dir = data_cfg.get("data_path", "../data_bc")
        splits_dir = "data/splits"

        # Load or create splits
        splits_path = Path(splits_dir)
        if splits_path.exists() and (splits_path / "train.json").exists():
            print("Loading existing splits...")
            train_data, val_data, test_data = load_splits(splits_dir)
        else:
            print("Creating new splits...")
            train_data, val_data, test_data = create_splits(
                data_dir,
                output_dir=splits_dir,
                seed=seed,
            )

        print(f"Data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Create heuristics
        heuristics = create_heuristics(config)
        print(f"Heuristics: {[h.name for h in heuristics]}")

        # Create detector
        automaton_cfg = config.get("automaton", {})
        reward_cfg = config.get("reward", {})

        # Extract automaton type and build config
        automaton_type = automaton_cfg.get("type", "LR-I")
        automaton_config = {
            k: v for k, v in automaton_cfg.items()
            if k not in ["type", "alpha"]  # alpha is passed separately
        }

        detector = LADetector(
            heuristics=heuristics,
            alpha=automaton_cfg.get("alpha", 0.1),
            reward_type=reward_cfg.get("type", "binary"),
            iou_threshold=reward_cfg.get("iou_threshold", 0.3),
            seed=seed,
            automaton_type=automaton_type,
            automaton_config=automaton_config,
        )

        print(f"Automaton: {automaton_type} (alpha={automaton_cfg.get('alpha', 0.1)})")

        # Run training
        print("\n=== Training ===")
        train_stats = run_training(detector, train_data, data_dir, config, tracker)

        # Save probability history
        tracker.log_probability_history(
            [p.tolist() for p in detector.automaton.probability_history],
            detector.heuristic_names,
        )

        # Run evaluation
        print("\n=== Evaluation ===")
        val_metrics = run_evaluation(detector, val_data, data_dir, "val")
        test_metrics = run_evaluation(detector, test_data, data_dir, "test")

        # Combine metrics
        all_metrics = {
            **train_stats,
            **val_metrics,
            **test_metrics,
            "final_entropy": detector.automaton.get_entropy(),
            "best_heuristic": detector.get_best_heuristic()[0],
            "detector_config": detector.get_config(),
        }

        # Log metrics
        tracker.log_metrics(all_metrics)

        # Print results
        print("\n=== Results ===")
        print(f"Best heuristic: {all_metrics['best_heuristic']}")
        print(f"Final probabilities: {all_metrics['final_probabilities']}")
        print(f"Final entropy: {all_metrics['final_entropy']:.4f}")
        print(f"Val IoU: {all_metrics['val_iou']:.4f} (+/- {all_metrics['val_iou_std']:.4f})")
        print(f"Test IoU: {all_metrics['test_iou']:.4f} (+/- {all_metrics['test_iou_std']:.4f})")

        # Compare with baselines if available
        baselines = load_baseline_results()
        if baselines:
            print_baseline_comparison(all_metrics['test_iou'], baselines)
            all_metrics['baseline_comparison'] = {
                'random_iou': baselines.get('random', {}).get('iou'),
                'oracle_iou': baselines.get('oracle', {}).get('iou'),
                'la_vs_random': all_metrics['test_iou'] - baselines.get('random', {}).get('iou', 0),
                'la_vs_oracle': all_metrics['test_iou'] - baselines.get('oracle', {}).get('iou', 0),
            }
        else:
            print("\n[Note] No baseline results found. Run 'make baselines' for comparison.")

        # Plot convergence
        plot_convergence(detector, tracker, config)

        # End experiment
        tracker.end_experiment(status="completed")

        print(f"\nExperiment completed. Results saved to {exp_dir}")

    except Exception as e:
        tracker.end_experiment(status="failed")
        raise e


if __name__ == "__main__":
    main()
