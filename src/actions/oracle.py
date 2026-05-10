"""
Oracle action chooser: given current ClusterManager state + GT assigns,
returns the single best next curation action.

Action priority (applied in order until a non-KEEP action is found):
  1. DISCARD  — cluster has no GT match (all noise spikes)
  2. SPLIT    — cluster contains spikes from ≥2 GT units (contaminated)
  3. MERGE    — two clusters share the same dominant GT unit
  4. KEEP     — cluster is clean and sufficiently large
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class OracleAction:
    """A single oracle-recommended curation action."""

    action_type: str           # "SPLIT" | "MERGE" | "DISCARD" | "KEEP"
    cluster_id: int
    target_id: Optional[int]   # Set for MERGE actions only
    gt_reasoning: str          # Human-readable justification grounded in GT


def choose_oracle_action(
    assigns: np.ndarray,
    gt_assigns: np.ndarray,
    active_clusters: Sequence[int],
    noise_gt_label: int = 0,
    purity_threshold: float = 0.80,
    min_spike_count: int = 500,
) -> OracleAction:
    """
    Return the highest-priority oracle action across all active clusters.

    Args:
        assigns: Current per-spike cluster labels (n_spikes,).
        gt_assigns: Ground-truth per-spike unit labels (n_spikes,).
            Label 0 is treated as noise/unmatched.
        active_clusters: List of currently active (non-discarded) cluster IDs.
        noise_gt_label: GT label that marks noise spikes (default 0).
        purity_threshold: A cluster is "clean" when its dominant GT unit
            accounts for ≥ this fraction of its spikes.
        min_spike_count: Clusters below this are DISCARD candidates first.

    Returns:
        The single highest-priority OracleAction.
    """
    active = list(active_clusters)

    # --- Pre-compute GT composition per cluster ---
    compositions: dict[int, dict[int, int]] = {}
    for cid in active:
        mask = assigns == cid
        gt_labels = gt_assigns[mask]
        unique, counts = np.unique(gt_labels, return_counts=True)
        compositions[cid] = dict(zip(unique.tolist(), counts.tolist()))

    # --- Pass 1: DISCARD (no GT match or too small and noisy) ---
    for cid in active:
        comp = compositions[cid]
        total = sum(comp.values())
        noise_count = comp.get(noise_gt_label, 0)
        if total == 0 or noise_count == total:
            return OracleAction(
                action_type="DISCARD",
                cluster_id=cid,
                target_id=None,
                gt_reasoning=(
                    f"Cluster {cid} contains {total} spikes, all labeled as noise "
                    f"(GT label {noise_gt_label}). No ground-truth unit is represented."
                ),
            )

    # --- Pass 2: SPLIT (dominant GT unit < purity_threshold) ---
    for cid in active:
        comp = compositions[cid]
        total = sum(v for k, v in comp.items() if k != noise_gt_label)
        if total == 0:
            continue
        dominant_gt, dominant_count = max(
            ((k, v) for k, v in comp.items() if k != noise_gt_label),
            key=lambda kv: kv[1],
        )
        purity = dominant_count / sum(comp.values())
        if purity < purity_threshold:
            gt_units = {k: v for k, v in comp.items() if k != noise_gt_label}
            unit_summary = ", ".join(
                f"GT-{k}: {v} spikes ({100*v/sum(comp.values()):.0f}%)"
                for k, v in sorted(gt_units.items(), key=lambda kv: -kv[1])
            )
            return OracleAction(
                action_type="SPLIT",
                cluster_id=cid,
                target_id=None,
                gt_reasoning=(
                    f"Cluster {cid} is contaminated (purity={purity:.2f} < {purity_threshold}). "
                    f"Dominant GT unit: {dominant_gt}. Composition: {unit_summary}."
                ),
            )

    # --- Pass 3: MERGE (two clusters share dominant GT unit) ---
    dominant_map: dict[int, list[int]] = {}   # gt_unit → [cluster_ids]
    for cid in active:
        comp = compositions[cid]
        non_noise = {k: v for k, v in comp.items() if k != noise_gt_label}
        if not non_noise:
            continue
        dominant_gt = max(non_noise, key=lambda k: non_noise[k])
        dominant_map.setdefault(dominant_gt, []).append(cid)

    for gt_unit, cluster_ids in dominant_map.items():
        if len(cluster_ids) >= 2:
            cid_a, cid_b = cluster_ids[0], cluster_ids[1]
            return OracleAction(
                action_type="MERGE",
                cluster_id=cid_a,
                target_id=cid_b,
                gt_reasoning=(
                    f"Clusters {cid_a} and {cid_b} both have GT unit {gt_unit} "
                    f"as their dominant unit — they represent the same neuron split "
                    "across two clusters."
                ),
            )

    # --- Pass 4: KEEP (all clusters are clean) ---
    return OracleAction(
        action_type="KEEP",
        cluster_id=active[0] if active else -1,
        target_id=None,
        gt_reasoning="All active clusters are sufficiently pure. No further action required.",
    )


def best_oracle_action_for_cluster(
    cluster_id: int,
    assigns: np.ndarray,
    gt_assigns: np.ndarray,
    active_clusters: Sequence[int],
    noise_gt_label: int = 0,
    purity_threshold: float = 0.80,
) -> OracleAction:
    """
    Return the oracle action specifically for one cluster_id.

    Used during trajectory generation when the student has already selected
    a cluster to act on and we need the oracle's verdict for that specific cluster.
    """
    mask = assigns == cluster_id
    if not np.any(mask):
        return OracleAction(
            action_type="DISCARD",
            cluster_id=cluster_id,
            target_id=None,
            gt_reasoning=f"Cluster {cluster_id} has no spikes.",
        )

    gt_labels = gt_assigns[mask]
    unique, counts = np.unique(gt_labels, return_counts=True)
    comp = dict(zip(unique.tolist(), counts.tolist()))
    total = sum(comp.values())
    noise_count = comp.get(noise_gt_label, 0)

    if noise_count == total:
        return OracleAction(
            action_type="DISCARD",
            cluster_id=cluster_id,
            target_id=None,
            gt_reasoning=f"Cluster {cluster_id}: all {total} spikes are noise.",
        )

    non_noise = {k: v for k, v in comp.items() if k != noise_gt_label}
    dominant_gt, dominant_count = max(non_noise.items(), key=lambda kv: kv[1])
    purity = dominant_count / total

    if purity < purity_threshold:
        return OracleAction(
            action_type="SPLIT",
            cluster_id=cluster_id,
            target_id=None,
            gt_reasoning=(
                f"Cluster {cluster_id}: purity={purity:.2f} < {purity_threshold}. "
                f"Dominant GT unit: {dominant_gt} ({dominant_count}/{total} spikes)."
            ),
        )

    # Check if another active cluster shares dominant GT unit → MERGE
    for other_cid in active_clusters:
        if other_cid == cluster_id:
            continue
        other_mask = assigns == other_cid
        other_gt = gt_assigns[other_mask]
        other_unique, other_counts = np.unique(other_gt, return_counts=True)
        other_comp = dict(zip(other_unique.tolist(), other_counts.tolist()))
        other_non_noise = {k: v for k, v in other_comp.items() if k != noise_gt_label}
        if not other_non_noise:
            continue
        other_dominant = max(other_non_noise, key=lambda k: other_non_noise[k])
        if other_dominant == dominant_gt:
            return OracleAction(
                action_type="MERGE",
                cluster_id=cluster_id,
                target_id=other_cid,
                gt_reasoning=(
                    f"Clusters {cluster_id} and {other_cid} share dominant GT unit "
                    f"{dominant_gt} — merge them."
                ),
            )

    return OracleAction(
        action_type="KEEP",
        cluster_id=cluster_id,
        target_id=None,
        gt_reasoning=(
            f"Cluster {cluster_id}: purity={purity:.2f}, dominant GT unit {dominant_gt}. Clean."
        ),
    )
