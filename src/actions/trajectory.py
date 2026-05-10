"""
Build a ground-truth action trajectory for a single channel.

Starting from the initial overclustered state stored in raw/, repeatedly queries
the oracle to determine the optimal next action, applies it to ClusterManager,
and records each step until all active clusters are KEEP.

Output: actions.jsonl  (one JSON object per line)
  {"step": int, "action_type": str, "cluster_id": int, "target_id": int|null,
   "gt_reasoning": str, "n_active_before": int, "n_active_after": int}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.cluster.manager import ClusterManager
from src.cluster.auto_filter import automatic_size_filter
from .oracle import choose_oracle_action, OracleAction


def build_gt_trajectory(
    raw_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    auto_discard_threshold: int = 500,
    purity_threshold: float = 0.80,
    max_steps: int = 200,
    force: bool = False,
) -> list[dict]:
    """
    Build the oracle-optimal action trajectory for one channel.

    Args:
        raw_dir: Channel raw/ directory containing numpy arrays.
        output_dir: Where to write actions.jsonl (defaults to raw_dir/../actions/).
        auto_discard_threshold: Clusters below this spike count are auto-discarded first.
        purity_threshold: Passed to choose_oracle_action.
        max_steps: Safety cap on trajectory length.
        force: Overwrite existing actions.jsonl.

    Returns:
        List of step dicts (same content as actions.jsonl lines).
    """
    raw_dir = Path(raw_dir)

    if output_dir is None:
        output_dir = raw_dir.parent / "actions"
    output_dir = Path(output_dir)

    out_file = output_dir / "actions.jsonl"
    if out_file.exists() and not force:
        print(f"  [skip] {out_file} already exists (use --force to overwrite)")
        steps = []
        with open(out_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
        return steps

    # --- Load arrays ---
    waveforms = np.load(raw_dir / "waveforms.npy")
    spike_times = np.load(raw_dir / "spike_times.npy")
    overcluster_assigns = np.load(raw_dir / "overcluster_assigns.npy")
    hierarchy_assigns = np.load(raw_dir / "hierarchy_assigns.npy")
    hierarchy_tree = np.load(raw_dir / "hierarchy_tree.npy")
    gt_assigns = np.load(raw_dir / "gt_assigns.npy")

    # --- Initialize manager ---
    manager = ClusterManager(
        initial_assigns=hierarchy_assigns,
        overcluster_assigns=overcluster_assigns,
        hierarchy_tree=hierarchy_tree,
        spike_times=spike_times,
        waveforms=waveforms,
    )

    # --- Phase 0: auto-discard tiny clusters ---
    _, filter_actions = automatic_size_filter(manager.assigns, auto_discard_threshold)
    for fa in filter_actions:
        manager.discard_cluster(fa.cluster_id)

    # --- Oracle trajectory loop ---
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    for step_idx in range(max_steps):
        active = manager.get_active_clusters()
        if not active:
            break

        action: OracleAction = choose_oracle_action(
            assigns=manager.assigns,
            gt_assigns=gt_assigns,
            active_clusters=active,
            purity_threshold=purity_threshold,
        )

        n_before = len(active)

        record = {
            "step": step_idx,
            "action_type": action.action_type,
            "cluster_id": action.cluster_id,
            "target_id": action.target_id,
            "gt_reasoning": action.gt_reasoning,
            "n_active_before": n_before,
        }

        if action.action_type == "KEEP":
            record["n_active_after"] = n_before
            steps.append(record)
            break

        _apply_action(manager, action)

        record["n_active_after"] = len(manager.get_active_clusters())
        steps.append(record)

    # --- Write output ---
    summary = {
        "n_steps": len(steps),
        "action_type_counts": _count_actions(steps),
    }

    with open(out_file, "w") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")

    with open(output_dir / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  [done] {out_file.parent.parent.name}/{out_file.parent.parent.parent.name}: "
        f"{len(steps)} steps  "
        f"({summary['action_type_counts']})"
    )
    return steps


def _apply_action(manager: ClusterManager, action: OracleAction) -> None:
    """Apply a single oracle action to the ClusterManager."""
    if action.action_type == "SPLIT":
        try:
            manager.split_last_merge(action.cluster_id)
        except Exception:
            manager.split_by_current_overclusters(action.cluster_id)
    elif action.action_type == "DISCARD":
        manager.discard_cluster(action.cluster_id)
    elif action.action_type == "MERGE" and action.target_id is not None:
        manager.merge_clusters([action.cluster_id, action.target_id], target_id=action.target_id)


def _count_actions(steps: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in steps:
        t = s["action_type"]
        counts[t] = counts.get(t, 0) + 1
    return counts
