"""
Channel sampler for adaptation experiments.

Splits the n_channels of a setting into train and eval sets,
then builds an HF-format dataset from training trajectories.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainEvalSplit:
    setting_id: str
    train_channel_ids: List[str]
    eval_channel_ids: List[str]
    seed: int


def sample_channels(
    setting_id: str,
    output_dir: str | Path,
    n_train: int,
    n_eval: int,
    seed: int = 0,
) -> TrainEvalSplit:
    """
    Sample train/eval channel splits for a setting.

    Discovers available channels by looking for trajectory.jsonl files
    under output/<setting_id>/<channel_id>/trajectory/.

    Args:
        setting_id: Setting identifier.
        output_dir: Root output directory.
        n_train: Number of channels for training.
        n_eval: Number of channels for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        TrainEvalSplit with channel IDs for train and eval.
    """
    setting_dir = Path(output_dir) / setting_id
    available = sorted(
        ch.name
        for ch in setting_dir.iterdir()
        if ch.is_dir() and (ch / "trajectory" / "trajectory.jsonl").exists()
    )

    if len(available) < n_train + n_eval:
        raise ValueError(
            f"Setting {setting_id} has {len(available)} channels with trajectories, "
            f"but n_train + n_eval = {n_train + n_eval}."
        )

    rng = random.Random(seed)
    selected = rng.sample(available, n_train + n_eval)
    train_ids = selected[:n_train]
    eval_ids = selected[n_train:]

    return TrainEvalSplit(
        setting_id=setting_id,
        train_channel_ids=train_ids,
        eval_channel_ids=eval_ids,
        seed=seed,
    )


def build_sft_dataset_from_split(
    split: TrainEvalSplit,
    output_dir: str | Path,
    adapter_run_id: str,
) -> Path:
    """
    Build a supervised fine-tuning dataset from train-split trajectories.

    Each trajectory step where gt_action != KEEP becomes one training example:
      {"prompt": "<images + context>", "completion": "<gt_action>\\n<gt_reasoning>"}

    The dataset is written as train_dataset.jsonl under
    output/<setting_id>/adapters/<adapter_run_id>/.

    Args:
        split: TrainEvalSplit from sample_channels().
        output_dir: Root output directory.
        adapter_run_id: Unique identifier for this adaptation run.

    Returns:
        Path to the adapter run directory.
    """
    from src.trajectories.record import load_trajectory

    adapter_dir = Path(output_dir) / split.setting_id / "adapters" / adapter_run_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_examples: list[dict] = []
    for ch_id in split.train_channel_ids:
        traj_file = Path(output_dir) / split.setting_id / ch_id / "trajectory" / "trajectory.jsonl"
        steps = load_trajectory(traj_file)
        for step in steps:
            if step.gt_action == "KEEP":
                continue
            train_examples.append({
                "channel_id": ch_id,
                "setting_id": split.setting_id,
                "step": step.step,
                "action_phase": step.action_phase,
                "cluster_id": step.cluster_id,
                "target_id": step.target_id,
                "gt_action": step.gt_action,
                "gt_reasoning": step.gt_reasoning,
                "teacher_feedback": step.teacher_feedback,
                "student_action": step.student_action,
                "student_rationale": step.student_rationale,
                "image_paths": step.image_paths,
            })

    with open(adapter_dir / "train_dataset.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Record eval channel IDs for Stage 5
    eval_meta = {
        "setting_id": split.setting_id,
        "eval_channel_ids": split.eval_channel_ids,
        "train_channel_ids": split.train_channel_ids,
        "n_train_examples": len(train_examples),
        "seed": split.seed,
    }
    with open(adapter_dir / "eval_dataset.jsonl", "w") as f:
        json.dump(eval_meta, f, indent=2)

    print(
        f"  [done] SFT dataset: {len(train_examples)} examples from "
        f"{len(split.train_channel_ids)} train channels → {adapter_dir}"
    )
    return adapter_dir
