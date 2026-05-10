"""
Step-level action accuracy metrics.

Computes per-step action prediction accuracy, confusion matrix, and
sequence-level edit distance to the GT action trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


ACTION_TYPES = ["SPLIT", "MERGE", "DISCARD", "KEEP", "NOT_MERGE"]


@dataclass
class ActionMetrics:
    """Per-channel step-level action prediction metrics."""

    channel_id: str
    setting_id: str
    n_steps: int
    overall_accuracy: float
    per_type_accuracy: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]   # confusion_matrix[gt][pred] = count
    edit_distance: int                             # Levenshtein distance to GT sequence

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "setting_id": self.setting_id,
            "n_steps": self.n_steps,
            "overall_accuracy": self.overall_accuracy,
            "per_type_accuracy": self.per_type_accuracy,
            "confusion_matrix": self.confusion_matrix,
            "edit_distance": self.edit_distance,
        }


def compute_action_metrics(
    predictions: List[dict],
    channel_id: str = "",
    setting_id: str = "",
) -> ActionMetrics:
    """
    Compute action metrics from a list of prediction dicts.

    Each dict must have keys: gt_action, student_action.

    Args:
        predictions: List of prediction dicts (from evaluator.py or trajectory.jsonl).
        channel_id: Channel identifier for the metrics object.
        setting_id: Setting identifier for the metrics object.

    Returns:
        ActionMetrics dataclass.
    """
    if not predictions:
        return ActionMetrics(
            channel_id=channel_id,
            setting_id=setting_id,
            n_steps=0,
            overall_accuracy=0.0,
            per_type_accuracy={},
            confusion_matrix={},
            edit_distance=0,
        )

    n_correct = 0
    type_correct: Dict[str, int] = {}
    type_total: Dict[str, int] = {}
    confusion: Dict[str, Dict[str, int]] = {}

    gt_seq: List[str] = []
    pred_seq: List[str] = []

    for p in predictions:
        gt = p["gt_action"].upper()
        pred = p.get("student_action", "").upper()

        gt_seq.append(gt)
        pred_seq.append(pred)

        if gt == pred:
            n_correct += 1
        type_correct[gt] = type_correct.get(gt, 0) + (1 if gt == pred else 0)
        type_total[gt] = type_total.get(gt, 0) + 1

        confusion.setdefault(gt, {})
        confusion[gt][pred] = confusion[gt].get(pred, 0) + 1

    per_type_acc = {
        t: type_correct.get(t, 0) / type_total[t]
        for t in type_total
    }

    return ActionMetrics(
        channel_id=channel_id,
        setting_id=setting_id,
        n_steps=len(predictions),
        overall_accuracy=n_correct / len(predictions),
        per_type_accuracy=per_type_acc,
        confusion_matrix=confusion,
        edit_distance=_levenshtein(gt_seq, pred_seq),
    )


def load_predictions_and_compute(
    eval_dir: str | Path,
    channel_id: str = "",
    setting_id: str = "",
) -> ActionMetrics:
    """
    Load eval_predictions.jsonl from eval_dir and compute ActionMetrics.
    """
    pred_file = Path(eval_dir) / "eval_predictions.jsonl"
    predictions: list[dict] = []
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return compute_action_metrics(predictions, channel_id=channel_id, setting_id=setting_id)


def _levenshtein(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute Levenshtein edit distance between two string sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
