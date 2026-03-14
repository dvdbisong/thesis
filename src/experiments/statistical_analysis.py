"""
Statistical Analysis Module for LA Experiments

Provides statistical tests for comparing Learning Automata algorithms:
- Paired t-tests for pairwise comparison
- Effect size (Cohen's d) for practical significance
- Friedman test for overall ranking
- Confidence intervals
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def paired_ttest(
    values1: List[float],
    values2: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform paired t-test between two sets of results.

    Args:
        values1: Results from algorithm 1
        values2: Results from algorithm 2
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, and significance
    """
    if len(values1) != len(values2):
        raise ValueError("Both lists must have same length (paired samples)")

    t_stat, p_value = stats.ttest_rel(values1, values2)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
        "n_samples": len(values1),
    }


def cohens_d(values1: List[float], values2: List[float]) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        values1: Results from algorithm 1
        values2: Results from algorithm 2

    Returns:
        Cohen's d value
    """
    diff = np.array(values1) - np.array(values2)
    d = np.mean(diff) / np.std(diff, ddof=1)
    return float(d)


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.

    Args:
        values: Sample values
        confidence: Confidence level (default 95%)

    Returns:
        (lower_bound, upper_bound)
    """
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (float(mean - h), float(mean + h))


def friedman_test(
    results: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform Friedman test for ranking multiple algorithms.

    Args:
        results: Dictionary mapping algorithm names to list of scores
        alpha: Significance level

    Returns:
        Dictionary with test statistic, p-value, and rankings
    """
    # Convert to matrix form (algorithms x samples)
    names = list(results.keys())
    data = np.array([results[name] for name in names])

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*data)

    # Compute mean ranks
    ranks = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        ranks[:, i] = stats.rankdata(-data[:, i])  # Negative for descending

    mean_ranks = np.mean(ranks, axis=1)

    # Create ranking
    ranking = sorted(zip(names, mean_ranks), key=lambda x: x[1])

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
        "mean_ranks": {name: float(rank) for name, rank in zip(names, mean_ranks)},
        "ranking": [(name, float(rank)) for name, rank in ranking],
    }


def compare_algorithms(
    experiment_results: Dict[str, Dict[str, Any]],
    metric: str = "test_iou",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Comprehensive comparison of multiple algorithms.

    Args:
        experiment_results: Dict mapping algorithm names to aggregated results
        metric: Metric to compare on
        alpha: Significance level

    Returns:
        Comprehensive comparison report
    """
    # Extract values for each algorithm
    algo_values = {}
    for name, result in experiment_results.items():
        individual_runs = result.get("individual_runs", [])
        values = [run.get(metric, 0) for run in individual_runs]
        if values:
            algo_values[name] = values

    if len(algo_values) < 2:
        return {"error": "Need at least 2 algorithms to compare"}

    names = list(algo_values.keys())

    # Summary statistics
    summary = {}
    for name, values in algo_values.items():
        ci_low, ci_high = confidence_interval(values)
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_95": (ci_low, ci_high),
            "n": len(values),
        }

    # Pairwise comparisons
    pairwise = {}
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            key = f"{name1}_vs_{name2}"
            ttest = paired_ttest(algo_values[name1], algo_values[name2], alpha)
            d = cohens_d(algo_values[name1], algo_values[name2])

            pairwise[key] = {
                **ttest,
                "cohens_d": d,
                "effect_size": effect_size_interpretation(d),
                "winner": name1 if np.mean(algo_values[name1]) > np.mean(algo_values[name2]) else name2,
            }

    # Friedman test for overall ranking
    friedman = friedman_test(algo_values, alpha)

    return {
        "metric": metric,
        "n_algorithms": len(names),
        "summary": summary,
        "pairwise_comparisons": pairwise,
        "friedman_test": friedman,
        "best_algorithm": friedman["ranking"][0][0],
    }


def generate_latex_table(comparison: Dict[str, Any]) -> str:
    """
    Generate LaTeX table from comparison results.

    Args:
        comparison: Output from compare_algorithms

    Returns:
        LaTeX table string
    """
    summary = comparison["summary"]
    metric = comparison["metric"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm Comparison: " + metric.replace("_", r"\_") + "}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{Mean} & \textbf{Std} & \textbf{95\% CI} \\",
        r"\midrule",
    ]

    # Sort by mean descending
    sorted_algos = sorted(summary.items(), key=lambda x: x[1]["mean"], reverse=True)

    for name, stats in sorted_algos:
        ci = stats["ci_95"]
        line = f"{name} & {stats['mean']:.4f} & {stats['std']:.4f} & [{ci[0]:.4f}, {ci[1]:.4f}] \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:" + comparison["metric"] + "}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def print_comparison_report(comparison: Dict[str, Any]) -> None:
    """Print formatted comparison report."""
    print(f"\n{'='*70}")
    print(f"STATISTICAL COMPARISON: {comparison['metric']}")
    print(f"{'='*70}\n")

    # Summary
    print("SUMMARY STATISTICS:")
    print("-" * 50)
    print(f"{'Algorithm':<20} {'Mean':>10} {'Std':>10} {'95% CI':>20}")
    print("-" * 50)

    sorted_algos = sorted(
        comparison["summary"].items(),
        key=lambda x: x[1]["mean"],
        reverse=True
    )

    for name, stats in sorted_algos:
        ci = stats["ci_95"]
        print(f"{name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    # Pairwise comparisons
    print(f"\n{'='*70}")
    print("PAIRWISE COMPARISONS:")
    print("-" * 70)

    for key, result in comparison["pairwise_comparisons"].items():
        sig = "***" if result["significant"] else ""
        print(f"\n{key}:")
        print(f"  t = {result['t_statistic']:.3f}, p = {result['p_value']:.4f} {sig}")
        print(f"  Cohen's d = {result['cohens_d']:.3f} ({result['effect_size']})")
        print(f"  Winner: {result['winner']}")

    # Friedman test
    print(f"\n{'='*70}")
    print("OVERALL RANKING (Friedman Test):")
    print("-" * 50)

    friedman = comparison["friedman_test"]
    sig = "***" if friedman["significant"] else ""
    print(f"Chi-square = {friedman['statistic']:.3f}, p = {friedman['p_value']:.4f} {sig}")
    print(f"\nRanking:")

    for i, (name, rank) in enumerate(friedman["ranking"], 1):
        print(f"  {i}. {name} (mean rank: {rank:.2f})")

    print(f"\n{'='*70}")
    print(f"BEST ALGORITHM: {comparison['best_algorithm']}")
    print(f"{'='*70}\n")


def load_aggregated_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all aggregated results from directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary mapping experiment names to results
    """
    results = {}
    results_path = Path(results_dir)

    for path in results_path.glob("*_aggregated.json"):
        with open(path) as f:
            data = json.load(f)
        name = data.get("name", path.stem.replace("_aggregated", ""))
        results[name] = data

    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing aggregated results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_iou",
        help="Metric to compare",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX table",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to compare (default: all)",
    )
    args = parser.parse_args()

    # Load results
    all_results = load_aggregated_results(args.results_dir)

    if not all_results:
        print(f"No aggregated results found in {args.results_dir}")
        return

    # Filter if specific experiments requested
    if args.experiments:
        all_results = {k: v for k, v in all_results.items() if k in args.experiments}

    if len(all_results) < 2:
        print("Need at least 2 experiments to compare")
        return

    # Run comparison
    comparison = compare_algorithms(all_results, args.metric, args.alpha)

    # Print report
    print_comparison_report(comparison)

    # Generate LaTeX if requested
    if args.latex:
        print("\nLaTeX Table:")
        print("-" * 50)
        print(generate_latex_table(comparison))

    # Save comparison
    output_path = Path(args.results_dir) / "comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {output_path}")


if __name__ == "__main__":
    main()
