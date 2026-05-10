"""
TrajectoryStep dataclass and .jsonl serialization.

Each step records everything needed to reconstruct the teacher-student
interaction at one point in the curation workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrajectoryStep:
    """One step of a teacher-student curation interaction."""

    step: int
    channel_id: str
    setting_id: str

    # --- Cluster context ---
    cluster_id: int
    action_phase: str              # "phase1" (split/discard) | "phase2" (merge)
    target_id: Optional[int]       # Populated for merge steps

    # --- Ground truth ---
    gt_action: str                 # "SPLIT" | "MERGE" | "DISCARD" | "KEEP"
    gt_reasoning: str

    # --- Student (zero-shot VLM) ---
    student_action: str            # Model's predicted action
    student_rationale: str         # Model's stated reasoning
    student_model: str

    # --- Teacher (GT-aware VLM) ---
    teacher_feedback: str
    teacher_model: str

    # --- Supporting files ---
    image_paths: List[str] = field(default_factory=list)   # Relative paths to images

    # --- Optional metadata ---
    n_spikes: Optional[int] = None
    n_active_clusters: Optional[int] = None


def save_trajectory(steps: List[TrajectoryStep], path: str | Path) -> None:
    """Write a list of TrajectorySteps to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for step in steps:
            f.write(json.dumps(asdict(step)) + "\n")


def load_trajectory(path: str | Path) -> List[TrajectoryStep]:
    """Load a .jsonl file into a list of TrajectorySteps."""
    steps: List[TrajectoryStep] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                steps.append(TrajectoryStep(**d))
    return steps


def trajectory_summary(steps: List[TrajectoryStep]) -> dict:
    """Return a summary dict for a trajectory."""
    action_counts: dict[str, int] = {}
    student_correct = 0
    for s in steps:
        action_counts[s.gt_action] = action_counts.get(s.gt_action, 0) + 1
        if s.student_action.upper() == s.gt_action.upper():
            student_correct += 1

    return {
        "n_steps": len(steps),
        "gt_action_counts": action_counts,
        "student_action_accuracy": student_correct / len(steps) if steps else 0.0,
    }
