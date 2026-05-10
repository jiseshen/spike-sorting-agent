"""
Evaluator: replay the student (optionally with a finetuned adapter) on
held-out channels and collect per-step predictions.

Writes per-channel eval_predictions.jsonl under:
  output/<setting_id>/<channel_id>/eval/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.cluster.manager import ClusterManager
from src.cluster.auto_filter import automatic_size_filter
from src.actions.oracle import best_oracle_action_for_cluster
from src.trajectories.student import StudentRunner


def evaluate_on_channels(
    setting_id: str,
    channel_ids: List[str],
    output_dir: str | Path,
    student: StudentRunner,
    small_cluster_threshold: int = 4000,
    auto_discard_threshold: int = 500,
    force: bool = False,
) -> dict[str, list[dict]]:
    """
    Run the student on held-out channels and record predictions vs GT.

    Args:
        setting_id: Setting identifier.
        channel_ids: List of channel IDs to evaluate.
        output_dir: Root output directory.
        student: Configured StudentRunner (may point to a finetuned model).
        small_cluster_threshold: Phase 2 threshold.
        auto_discard_threshold: Phase 0 auto-discard threshold.
        force: Re-run even if predictions exist.

    Returns:
        Dict mapping channel_id → list of prediction dicts.
    """
    output_dir = Path(output_dir)
    all_predictions: dict[str, list[dict]] = {}

    for channel_id in channel_ids:
        raw_dir = output_dir / setting_id / channel_id / "raw"
        eval_dir = output_dir / setting_id / channel_id / "eval"
        pred_file = eval_dir / "eval_predictions.jsonl"

        if pred_file.exists() and not force:
            print(f"  [skip] {channel_id} predictions exist")
            preds = []
            with open(pred_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        preds.append(json.loads(line))
            all_predictions[channel_id] = preds
            continue

        if not (raw_dir / "waveforms.npy").exists():
            print(f"  [warn] Raw arrays missing for {channel_id}, skipping.")
            continue

        waveforms = np.load(raw_dir / "waveforms.npy")
        spike_times = np.load(raw_dir / "spike_times.npy")
        overcluster_assigns = np.load(raw_dir / "overcluster_assigns.npy")
        hierarchy_assigns = np.load(raw_dir / "hierarchy_assigns.npy")
        hierarchy_tree = np.load(raw_dir / "hierarchy_tree.npy")
        gt_assigns = np.load(raw_dir / "gt_assigns.npy")

        with open(raw_dir / "metadata.json") as f:
            meta = json.load(f)
        Fs = float(meta["Fs"])

        manager = ClusterManager(
            initial_assigns=hierarchy_assigns,
            overcluster_assigns=overcluster_assigns,
            hierarchy_tree=hierarchy_tree,
            spike_times=spike_times,
            waveforms=waveforms,
        )

        _, filter_actions = automatic_size_filter(manager.assigns, auto_discard_threshold)
        for fa in filter_actions:
            manager.discard_cluster(fa.cluster_id)

        predictions: list[dict] = []
        step_idx = 0

        # Phase 1
        for cid in list(manager.get_active_clusters()):
            if cid not in manager.get_active_clusters():
                continue

            student_resp = student.decide_phase1(
                cluster_id=cid,
                manager=manager,
                sampling_rate=Fs,
                output_dir=eval_dir / "student_inputs",
                step=step_idx,
            )

            oracle = best_oracle_action_for_cluster(
                cluster_id=cid,
                assigns=manager.assigns,
                gt_assigns=gt_assigns,
                active_clusters=manager.get_active_clusters(),
            )

            predictions.append({
                "step": step_idx,
                "phase": "phase1",
                "cluster_id": cid,
                "target_id": None,
                "gt_action": oracle.action_type,
                "student_action": student_resp.get("action", ""),
                "student_rationale": student_resp.get("rationale", ""),
                "correct": student_resp.get("action", "").upper() == oracle.action_type.upper(),
            })
            step_idx += 1

            # Advance state with GT action
            if oracle.action_type == "SPLIT":
                try:
                    manager.split_last_merge(cid)
                except Exception:
                    manager.split_by_current_overclusters(cid)
            elif oracle.action_type == "DISCARD":
                manager.discard_cluster(cid)

        # Phase 2
        active = manager.get_active_clusters()
        large_clusters = [c for c in active if manager.get_cluster_info(c)["n_spikes"] >= small_cluster_threshold]
        small_clusters = [c for c in active if manager.get_cluster_info(c)["n_spikes"] < small_cluster_threshold]

        for small_cid in small_clusters:
            if small_cid not in manager.get_active_clusters():
                continue
            for large_cid in large_clusters:
                if large_cid not in manager.get_active_clusters():
                    continue

                student_resp = student.decide_phase2(
                    small_cluster_id=small_cid,
                    large_cluster_id=large_cid,
                    manager=manager,
                    sampling_rate=Fs,
                    output_dir=eval_dir / "student_inputs",
                    step=step_idx,
                )

                oracle = best_oracle_action_for_cluster(
                    cluster_id=small_cid,
                    assigns=manager.assigns,
                    gt_assigns=gt_assigns,
                    active_clusters=manager.get_active_clusters(),
                )
                gt_action_type = "MERGE" if (oracle.action_type == "MERGE" and oracle.target_id == large_cid) else "NOT_MERGE"

                predictions.append({
                    "step": step_idx,
                    "phase": "phase2",
                    "cluster_id": small_cid,
                    "target_id": large_cid,
                    "gt_action": gt_action_type,
                    "student_action": student_resp.get("action", ""),
                    "student_rationale": student_resp.get("rationale", ""),
                    "correct": student_resp.get("action", "").upper() == gt_action_type.upper(),
                })
                step_idx += 1

                if gt_action_type == "MERGE":
                    manager.merge_clusters([small_cid, large_cid], target_id=large_cid)
                    break

        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(pred_file, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        all_predictions[channel_id] = predictions
        n_correct = sum(1 for p in predictions if p["correct"])
        print(
            f"  [done] {channel_id}: {len(predictions)} steps, "
            f"accuracy={n_correct/len(predictions):.2f}" if predictions else f"  [done] {channel_id}: 0 steps"
        )

    return all_predictions
