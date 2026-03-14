"""
Enhanced Experiment Runner with Multi-Seed Support

Provides batch execution of experiments with multiple random seeds,
automatic aggregation of results, and statistical analysis.
"""

import argparse
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from code.experiments.tracker import ExperimentTracker, load_config
from code.experiments.run_experiment import (
    create_heuristics,
    run_training,
    run_evaluation,
    plot_convergence,
    load_baseline_results,
)
from code.la_framework.detector import LADetector
from code.preprocessing.data_loader import create_splits, load_splits


# Standard seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]


class ExperimentRunner:
    """
    Runs experiments with multiple seeds and aggregates results.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        experiments_dir: str = "experiments",
        results_dir: str = "results",
    ):
        """
        Initialize the experiment runner.

        Args:
            config_path: Path to base configuration file
            experiments_dir: Directory for experiment outputs
            results_dir: Directory for aggregated results
        """
        self.config_path = config_path
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_single_experiment(
        self,
        config: Dict[str, Any],
        seed: int,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single experiment with a specific seed.

        Args:
            config: Configuration dictionary
            seed: Random seed
            experiment_name: Optional name for the experiment

        Returns:
            Dictionary of metrics
        """
        # Override seed in config
        config = copy.deepcopy(config)
        config["experiment"]["seed"] = seed

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create tracker
        name = experiment_name or config.get("experiment", {}).get("name", "experiment")
        tracker = ExperimentTracker(
            experiments_dir=str(self.experiments_dir),
            experiment_name=f"{name}_seed{seed}",
        )
        exp_dir = tracker.start_experiment(config)

        try:
            # Load data
            data_cfg = config.get("data", {})
            data_dir = data_cfg.get("data_path", "../data_bc")
            splits_dir = "data/splits"

            splits_path = Path(splits_dir)
            if splits_path.exists() and (splits_path / "train.json").exists():
                train_data, val_data, test_data = load_splits(splits_dir)
            else:
                train_data, val_data, test_data = create_splits(
                    data_dir, output_dir=splits_dir, seed=seed
                )

            # Create heuristics
            heuristics = create_heuristics(config)

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
                reward_type=reward_cfg.get("type", "continuous"),
                iou_threshold=reward_cfg.get("iou_threshold", 0.3),
                seed=seed,
                automaton_type=automaton_type,
                automaton_config=automaton_config,
            )

            # Run training
            train_stats = run_training(detector, train_data, data_dir, config, tracker)

            # Run evaluation
            val_metrics = run_evaluation(detector, val_data, data_dir, "val")
            test_metrics = run_evaluation(detector, test_data, data_dir, "test")

            # Combine metrics
            all_metrics = {
                **train_stats,
                **val_metrics,
                **test_metrics,
                "seed": seed,
                "final_entropy": detector.automaton.get_entropy(),
                "best_heuristic": detector.get_best_heuristic()[0],
                "final_probabilities": detector.get_probabilities(),
            }

            # Log and save
            tracker.log_metrics(all_metrics)
            plot_convergence(detector, tracker, config)
            tracker.end_experiment(status="completed")

            return all_metrics

        except Exception as e:
            tracker.end_experiment(status="failed")
            raise e

    def run_multi_seed(
        self,
        config: Dict[str, Any],
        seeds: List[int] = None,
        experiment_name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run experiment with multiple seeds and aggregate results.

        Args:
            config: Configuration dictionary
            seeds: List of random seeds (default: DEFAULT_SEEDS)
            experiment_name: Name for the experiment group
            verbose: Print progress

        Returns:
            Aggregated results with mean, std, and individual runs
        """
        seeds = seeds or DEFAULT_SEEDS
        name = experiment_name or config.get("experiment", {}).get("name", "experiment")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {name} with {len(seeds)} seeds: {seeds}")
            print(f"{'='*60}\n")

        # Run experiments
        all_results = []
        for seed in tqdm(seeds, desc=f"Seeds for {name}", disable=not verbose):
            if verbose:
                print(f"\n--- Seed {seed} ---")
            result = self.run_single_experiment(config, seed, name)
            all_results.append(result)

        # Aggregate results
        aggregated = self._aggregate_results(all_results, name)

        # Save aggregated results
        output_path = self.results_dir / f"{name}_aggregated.json"
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        if verbose:
            self._print_summary(aggregated)

        return aggregated

    def _aggregate_results(
        self, results: List[Dict[str, Any]], name: str
    ) -> Dict[str, Any]:
        """
        Aggregate results across multiple seeds.

        Args:
            results: List of result dictionaries
            name: Experiment name

        Returns:
            Aggregated statistics
        """
        # Metrics to aggregate
        numeric_metrics = [
            "test_iou", "test_f1", "test_precision", "test_recall",
            "val_iou", "val_f1", "val_precision", "val_recall",
            "mean_iou", "mean_reward", "final_entropy",
        ]

        aggregated = {
            "name": name,
            "n_seeds": len(results),
            "seeds": [r["seed"] for r in results],
            "timestamp": datetime.now().isoformat(),
            "individual_runs": results,
        }

        # Compute mean and std for numeric metrics
        for metric in numeric_metrics:
            values = [r.get(metric) for r in results if r.get(metric) is not None]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
                aggregated[f"{metric}_min"] = float(np.min(values))
                aggregated[f"{metric}_max"] = float(np.max(values))

        # Count best heuristic selections
        best_heuristics = [r.get("best_heuristic") for r in results]
        heuristic_counts = {}
        for h in best_heuristics:
            heuristic_counts[h] = heuristic_counts.get(h, 0) + 1
        aggregated["best_heuristic_counts"] = heuristic_counts
        aggregated["most_common_best"] = max(heuristic_counts, key=heuristic_counts.get)

        return aggregated

    def _print_summary(self, aggregated: Dict[str, Any]) -> None:
        """Print summary of aggregated results."""
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS: {aggregated['name']}")
        print(f"Seeds: {aggregated['n_seeds']}")
        print(f"{'='*60}")

        print(f"\nTest IoU: {aggregated.get('test_iou_mean', 0):.4f} "
              f"± {aggregated.get('test_iou_std', 0):.4f}")
        print(f"Test F1:  {aggregated.get('test_f1_mean', 0):.4f} "
              f"± {aggregated.get('test_f1_std', 0):.4f}")
        print(f"\nBest heuristic distribution: {aggregated.get('best_heuristic_counts', {})}")
        print(f"Most common: {aggregated.get('most_common_best', 'N/A')}")

        # Compare with baselines if available
        baselines = load_baseline_results()
        if baselines:
            test_iou = aggregated.get("test_iou_mean", 0)
            random_iou = baselines.get("random", {}).get("iou", 0)
            oracle_iou = baselines.get("oracle", {}).get("iou", 0)

            print(f"\nComparison:")
            print(f"  vs Random: {test_iou - random_iou:+.4f}")
            print(f"  vs Oracle: {test_iou - oracle_iou:+.4f} "
                  f"({(1 - test_iou/oracle_iou)*100:.1f}% gap)")

    def compare_experiments(
        self, experiment_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_names: List of experiment names to compare

        Returns:
            DataFrame with comparison
        """
        results = []

        for name in experiment_names:
            path = self.results_dir / f"{name}_aggregated.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                results.append({
                    "name": name,
                    "test_iou_mean": data.get("test_iou_mean", 0),
                    "test_iou_std": data.get("test_iou_std", 0),
                    "test_f1_mean": data.get("test_f1_mean", 0),
                    "n_seeds": data.get("n_seeds", 0),
                    "most_common_best": data.get("most_common_best", "N/A"),
                })

        df = pd.DataFrame(results)
        df = df.sort_values("test_iou_mean", ascending=False)
        return df


def run_config_sweep(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    seeds: List[int] = None,
    experiments_dir: str = "experiments",
    results_dir: str = "results",
) -> List[Dict[str, Any]]:
    """
    Run a parameter sweep over multiple configurations.

    Args:
        base_config: Base configuration to modify
        param_grid: Dictionary mapping param paths to lists of values
        seeds: Random seeds for each config
        experiments_dir: Directory for outputs
        results_dir: Directory for results

    Returns:
        List of aggregated results
    """
    from itertools import product

    runner = ExperimentRunner(
        experiments_dir=experiments_dir,
        results_dir=results_dir,
    )

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    all_results = []

    for combo in combinations:
        # Create config for this combination
        config = copy.deepcopy(base_config)

        # Set parameters
        name_parts = []
        for name, value in zip(param_names, combo):
            # Navigate to nested key
            keys = name.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
            name_parts.append(f"{keys[-1]}={value}")

        # Generate experiment name
        exp_name = "_".join(name_parts)

        # Run with multiple seeds
        result = runner.run_multi_seed(config, seeds, exp_name)
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run experiments with multiple seeds")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds to use",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of seeds (uses first N from default list)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Handle seeds
    seeds = args.seeds
    if args.n_seeds:
        seeds = DEFAULT_SEEDS[:args.n_seeds]

    # Run
    runner = ExperimentRunner()
    runner.run_multi_seed(config, seeds, args.name)


if __name__ == "__main__":
    main()
