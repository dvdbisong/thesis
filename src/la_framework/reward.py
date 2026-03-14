"""
Reward Functions for Learning Automata

Computes rewards based on the quality of predictions compared to ground truth.
The primary metric is IoU (Intersection over Union), also known as Jaccard Index.
"""

from typing import Dict, Tuple

import torch


def compute_iou(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) between prediction and ground truth.

    IoU = TP / (TP + FP + FN)

    Args:
        prediction: Binary prediction mask of shape (H, W)
        ground_truth: Binary ground truth mask of shape (H, W)

    Returns:
        IoU score in range [0, 1]
    """
    # Ensure binary tensors
    pred = (prediction > 0.5).float()
    gt = (ground_truth > 0.5).float()

    # Compute intersection and union
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = (intersection / union).item()
    return iou


def compute_metrics(
    prediction: torch.Tensor, ground_truth: torch.Tensor
) -> Dict[str, float]:
    """
    Compute multiple segmentation metrics.

    Args:
        prediction: Binary prediction mask of shape (H, W)
        ground_truth: Binary ground truth mask of shape (H, W)

    Returns:
        Dictionary with iou, precision, recall, f1, accuracy
    """
    # Ensure binary tensors
    pred = (prediction > 0.5).float()
    gt = (ground_truth > 0.5).float()

    # Compute confusion matrix elements
    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()

    # Compute metrics with epsilon to avoid division by zero
    eps = 1e-10

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class RewardFunction:
    """
    Base reward function for learning automata.

    Converts prediction quality (IoU) into a reward signal for the automaton.
    """

    def __init__(
        self,
        reward_type: str = "binary",
        iou_threshold: float = 0.3,
    ):
        """
        Initialize the reward function.

        Args:
            reward_type: "binary" (1 if IoU > threshold, else 0) or
                        "continuous" (use IoU directly as reward)
            iou_threshold: Threshold for binary reward
        """
        self.reward_type = reward_type
        self.iou_threshold = iou_threshold

    def compute_reward(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the reward for a prediction.

        Args:
            prediction: Binary prediction mask of shape (H, W)
            ground_truth: Binary ground truth mask of shape (H, W)

        Returns:
            Tuple of (reward, metrics_dict)
        """
        metrics = compute_metrics(prediction, ground_truth)
        iou = metrics["iou"]

        if self.reward_type == "binary":
            reward = 1.0 if iou > self.iou_threshold else 0.0
        elif self.reward_type == "continuous":
            reward = iou
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        return reward, metrics

    def get_config(self) -> Dict:
        return {
            "reward_type": self.reward_type,
            "iou_threshold": self.iou_threshold,
        }
