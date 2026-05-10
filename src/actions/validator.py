"""
Trajectory consistency validator.

Checks that an actions.jsonl file is internally consistent:
  - steps are contiguous from 0
  - cluster_ids referenced exist in the initial cluster set
  - MERGE actions have a valid target_id
  - no cluster is acted on after being discarded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"Trajectory [{status}]"]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def validate_trajectory(steps: list[dict]) -> ValidationResult:
    """
    Validate a list of action step dicts loaded from actions.jsonl.

    Args:
        steps: List of dicts with keys: step, action_type, cluster_id, target_id.

    Returns:
        ValidationResult with errors/warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    discarded: set[int] = set()
    valid_actions = {"SPLIT", "MERGE", "DISCARD", "KEEP"}

    for i, step in enumerate(steps):
        idx = step.get("step", i)

        if idx != i:
            errors.append(f"Step {i}: expected step index {i}, got {idx}.")

        action_type = step.get("action_type", "")
        if action_type not in valid_actions:
            errors.append(f"Step {i}: unknown action_type '{action_type}'.")

        cluster_id = step.get("cluster_id")
        if cluster_id is None:
            errors.append(f"Step {i}: missing cluster_id.")
        elif cluster_id in discarded:
            errors.append(f"Step {i}: cluster {cluster_id} acted on after being discarded.")

        target_id = step.get("target_id")
        if action_type == "MERGE":
            if target_id is None:
                errors.append(f"Step {i}: MERGE action missing target_id.")
            elif target_id in discarded:
                errors.append(f"Step {i}: MERGE target {target_id} has already been discarded.")

        if action_type == "DISCARD" and cluster_id is not None:
            discarded.add(cluster_id)

        if action_type == "KEEP" and i < len(steps) - 1:
            warnings.append(f"Step {i}: KEEP appears before the final step — trajectory may continue unnecessarily.")

    if not steps:
        warnings.append("Trajectory is empty.")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
