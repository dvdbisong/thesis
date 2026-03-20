"""
Baseline Methods for Comparison

Implements baseline heuristic selection strategies:
1. Random: Select heuristic uniformly at random for each tile
2. Fixed: Always use a single fixed heuristic
3. Oracle: Always select the best heuristic per tile (upper bound)

These baselines help evaluate the effectiveness of the LA approach.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.experiments.tracker import load_config
from src.experiments.run_experiment import create_heuristics
from src.la_framework.reward import compute_metrics
from src.preprocessing.data_loader import KelpTileDataset, load_splits, create_splits


def evaluate_random_baseline(
    heuristics: List,
    dataset: KelpTileDataset,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate random heuristic selection.

    Args:
        heuristics: List of heuristics
        dataset: Dataset to evaluate on
        seed: Random seed

    Returns:
        Evaluation metrics
    """
    rng = np.random.RandomState(seed)
    all_metrics = []
    action_counts = {h.name: 0 for h in heuristics}

    for idx in tqdm(range(len(dataset)), desc="Random baseline"):
        tile, mask, _ = dataset[idx]

        # Select random heuristic
        action = rng.randint(len(heuristics))
        heuristic = heuristics[action]
        action_counts[heuristic.name] += 1

        # Get prediction and compute metrics
        pred = heuristic.predict(tile)
        metrics = compute_metrics(pred, mask)
        all_metrics.append(metrics)

    return {
        "name": "random",
        "iou": np.mean([m["iou"] for m in all_metrics]),
        "iou_std": np.std([m["iou"] for m in all_metrics]),
        "f1": np.mean([m["f1"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "action_counts": action_counts,
    }


def evaluate_fixed_baseline(
    heuristic,
    dataset: KelpTileDataset,
) -> Dict[str, Any]:
    """
    Evaluate using a single fixed heuristic.

    Args:
        heuristic: The heuristic to use
        dataset: Dataset to evaluate on

    Returns:
        Evaluation metrics
    """
    all_metrics = []

    for idx in tqdm(range(len(dataset)), desc=f"Fixed ({heuristic.name})"):
        tile, mask, _ = dataset[idx]

        pred = heuristic.predict(tile)
        metrics = compute_metrics(pred, mask)
        all_metrics.append(metrics)

    return {
        "name": f"fixed_{heuristic.name}",
        "heuristic": heuristic.name,
        "iou": np.mean([m["iou"] for m in all_metrics]),
        "iou_std": np.std([m["iou"] for m in all_metrics]),
        "f1": np.mean([m["f1"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
    }


def evaluate_oracle_baseline(
    heuristics: List,
    dataset: KelpTileDataset,
) -> Dict[str, Any]:
    """
    Evaluate oracle selection (best heuristic per tile).

    This is an upper bound on what any selection strategy can achieve.

    Args:
        heuristics: List of heuristics
        dataset: Dataset to evaluate on

    Returns:
        Evaluation metrics
    """
    all_metrics = []
    best_counts = {h.name: 0 for h in heuristics}

    for idx in tqdm(range(len(dataset)), desc="Oracle baseline"):
        tile, mask, _ = dataset[idx]

        # Try all heuristics and pick the best
        best_iou = -1
        best_metrics = None
        best_heuristic = None

        for heuristic in heuristics:
            pred = heuristic.predict(tile)
            metrics = compute_metrics(pred, mask)

            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_metrics = metrics
                best_heuristic = heuristic.name

        all_metrics.append(best_metrics)
        best_counts[best_heuristic] += 1

    return {
        "name": "oracle",
        "iou": np.mean([m["iou"] for m in all_metrics]),
        "iou_std": np.std([m["iou"] for m in all_metrics]),
        "f1": np.mean([m["f1"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "best_heuristic_counts": best_counts,
    }


def run_all_baselines(
    config: Dict[str, Any],
    test_data: List[Dict],
    data_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline evaluations.

    Args:
        config: Configuration dictionary
        test_data: Test data paths
        data_dir: Data directory

    Returns:
        Dictionary of baseline results
    """
    # Create heuristics
    heuristics = create_heuristics(config)

    # Set up ensemble if present
    for h in heuristics:
        if h.name == "ensemble":
            other_heuristics = [hh for hh in heuristics if hh.name != "ensemble"]
            h.set_heuristics(other_heuristics)

    # Create dataset
    dataset = KelpTileDataset(data_dir, tile_paths=test_data)

    results = {}

    # Random baseline
    print("\n=== Random Baseline ===")
    seed = config.get("experiment", {}).get("seed", 42)
    results["random"] = evaluate_random_baseline(heuristics, dataset, seed)
    print(f"IoU: {results['random']['iou']:.4f} (+/- {results['random']['iou_std']:.4f})")

    # Fixed baselines (one per heuristic)
    print("\n=== Fixed Baselines ===")
    for h in heuristics:
        result = evaluate_fixed_baseline(h, dataset)
        results[result["name"]] = result
        print(f"{h.name}: IoU = {result['iou']:.4f} (+/- {result['iou_std']:.4f})")

    # Oracle baseline
    print("\n=== Oracle Baseline ===")
    results["oracle"] = evaluate_oracle_baseline(heuristics, dataset)
    print(f"IoU: {results['oracle']['iou']:.4f} (+/- {results['oracle']['iou_std']:.4f})")
    print(f"Best heuristic distribution: {results['oracle']['best_heuristic_counts']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baselines.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Set random seed
    seed = config.get("experiment", {}).get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_path", "../data_bc")
    splits_dir = "data/splits"

    splits_path = Path(splits_dir)
    if splits_path.exists() and (splits_path / "train.json").exists():
        print("Loading existing splits...")
        train_data, val_data, test_data = load_splits(splits_dir)
    else:
        print("Creating new splits...")
        train_data, val_data, test_data = create_splits(
            data_dir, output_dir=splits_dir, seed=seed
        )

    print(f"Test set: {len(test_data)} tiles")

    # Run baselines
    results = run_all_baselines(config, test_data, data_dir)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Method':<20} {'IoU':<10} {'F1':<10}")
    print("-" * 40)
    for name, result in sorted(results.items(), key=lambda x: x[1]["iou"], reverse=True):
        print(f"{name:<20} {result['iou']:.4f}     {result['f1']:.4f}")


if __name__ == "__main__":
    main()
