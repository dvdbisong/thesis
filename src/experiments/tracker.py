"""
Experiment Tracker

Manages experiment metadata, configuration, and results logging.
Creates timestamped experiment directories with all artifacts.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ExperimentTracker:
    """Tracks experiments with metadata, configs, and results."""

    def __init__(
        self,
        experiments_dir: str = "experiments",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Base directory for experiments
            experiment_name: Optional custom name for the experiment
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.experiments_dir / "index.json"
        self.experiment_name = experiment_name
        self.experiment_dir: Optional[Path] = None
        self.experiment_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []

    def start_experiment(self, config: Dict[str, Any]) -> Path:
        """
        Start a new experiment.

        Args:
            config: Experiment configuration dictionary

        Returns:
            Path to the experiment directory
        """
        self.start_time = datetime.now()
        self.config = config

        # Generate experiment ID
        exp_num = self._get_next_experiment_number()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        if self.experiment_name:
            self.experiment_id = f"exp_{exp_num:03d}_{self.experiment_name}_{timestamp}"
        else:
            name = config.get("experiment", {}).get("name", "unnamed")
            self.experiment_id = f"exp_{exp_num:03d}_{name}_{timestamp}"

        # Create experiment directory
        self.experiment_dir = self.experiments_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)

        # Save config
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Initialize metrics file
        self._save_metrics()

        print(f"Started experiment: {self.experiment_id}")
        print(f"Output directory: {self.experiment_dir}")

        return self.experiment_dir

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log data during experiment.

        Args:
            data: Dictionary of values to log
            step: Optional step/iteration number
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **data,
        }
        self.logs.append(entry)

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log final metrics.

        Args:
            metrics: Dictionary of metric values
        """
        self.metrics.update(metrics)
        self._save_metrics()

    def log_probability_history(self, history: List[List[float]], heuristic_names: List[str]):
        """
        Log probability vector history for convergence analysis.

        Args:
            history: List of probability vectors over time
            heuristic_names: Names of heuristics
        """
        self.metrics["probability_history"] = {
            "heuristics": heuristic_names,
            "values": history,
        }
        self._save_metrics()

    def save_plot(self, fig, name: str, format: str = "png", dpi: int = 150):
        """
        Save a matplotlib figure to the plots directory.

        Args:
            fig: Matplotlib figure object
            name: Name for the plot file (without extension)
            format: Image format (png, pdf, etc.)
            dpi: Resolution for raster formats
        """
        if self.experiment_dir is None:
            raise RuntimeError("Experiment not started. Call start_experiment() first.")

        plot_path = self.experiment_dir / "plots" / f"{name}.{format}"
        fig.savefig(plot_path, format=format, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot: {plot_path}")

    def end_experiment(self, status: str = "completed"):
        """
        End the experiment and finalize logging.

        Args:
            status: Experiment status (completed, failed, interrupted)
        """
        if self.experiment_dir is None:
            raise RuntimeError("Experiment not started.")

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # Update metrics with final info
        self.metrics["status"] = status
        self.metrics["duration_seconds"] = duration
        self.metrics["start_time"] = self.start_time.isoformat()
        self.metrics["end_time"] = end_time.isoformat()

        # Save final metrics
        self._save_metrics()

        # Save logs
        logs_path = self.experiment_dir / "logs.json"
        with open(logs_path, "w") as f:
            json.dump(self.logs, f, indent=2)

        # Update index
        self._update_index(status)

        print(f"Experiment {status}: {self.experiment_id}")
        print(f"Duration: {duration:.1f} seconds")

    def _get_next_experiment_number(self) -> int:
        """Get the next experiment number from the index."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                index = json.load(f)
            return len(index) + 1
        return 1

    def _save_metrics(self):
        """Save metrics to file."""
        if self.experiment_dir is None:
            return

        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _update_index(self, status: str):
        """Update the experiment index file."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                index = json.load(f)
        else:
            index = []

        entry = {
            "id": self.experiment_id,
            "path": str(self.experiment_dir),
            "name": self.config.get("experiment", {}).get("name", "unnamed"),
            "status": status,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": self.metrics.get("duration_seconds"),
        }

        # Add key metrics summary
        for key in ["test_iou", "test_f1", "final_entropy"]:
            if key in self.metrics:
                entry[key] = self.metrics[key]

        index.append(entry)

        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
