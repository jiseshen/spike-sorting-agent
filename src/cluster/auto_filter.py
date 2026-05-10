"""
Automated pre-filtering utilities for spike sorting curation.

These functions implement rule-based filtering that doesn't require LLM analysis.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class FilterAction:
    """Record of an automated filtering action."""
    cluster_id: int
    action: str  # 'm 0 X' for discard
    reasoning: str
    n_spikes: int


def automatic_size_filter(assigns: np.ndarray, 
                         threshold: int = 500) -> Tuple[List[int], List[FilterAction]]:
    """
    Automatically discard all clusters with < threshold spikes.
    
    This is Phase 0 filtering based on the principle that after hierarchical
    similarity-based clustering, small clusters are remnants/noise.
    
    Args:
        assigns: (n_spikes,) cluster assignments
        threshold: Minimum spike count (default: 500)
    
    Returns:
        kept_clusters: List of cluster IDs that passed filtering
        actions: List of FilterAction records for discarded clusters
    """
    unique_clusters = np.unique(assigns)
    # Exclude cluster 0 (already noise)
    unique_clusters = unique_clusters[unique_clusters != 0]
    
    kept_clusters = []
    actions = []
    
    for cluster_id in unique_clusters:
        n_spikes = (assigns == cluster_id).sum()
        
        if n_spikes < threshold:
            # Discard
            action = FilterAction(
                cluster_id=int(cluster_id),
                action=f"m 0 {cluster_id}",
                reasoning="Event count is too low for individual clusters at end of hierarchical merging.",
                n_spikes=int(n_spikes)
            )
            actions.append(action)
        else:
            # Keep for further analysis
            kept_clusters.append(int(cluster_id))
    
    return kept_clusters, actions


def generate_action_log(actions: List[FilterAction], 
                       output_path: str = None) -> str:
    """
    Generate MATLAB-style action log from filter actions.
    
    Args:
        actions: List of FilterAction records
        output_path: Optional file path to save log
    
    Returns:
        Log string in MATLAB format
    """
    lines = ["Actions,Action Reasoning"]
    
    for action in actions:
        # MATLAB format: 'm 0 X', 'Reasoning text'
        line = f"{action.action},\"{action.reasoning}\""
        lines.append(line)
    
    log_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(log_str)
    
    return log_str


# =====================================================================
# Convenience function for full Phase 0 filtering
# =====================================================================

def phase0_automatic_filtering(assigns: np.ndarray,
                               size_threshold: int = 500,
                               log_path: str = None) -> Tuple[List[int], str]:
    """
    Complete Phase 0 automatic filtering workflow.
    
    Args:
        assigns: (n_spikes,) cluster assignments
        size_threshold: Minimum spike count
        log_path: Optional path to save action log
    
    Returns:
        kept_clusters: List of cluster IDs passing to Phase 1
        action_log: CSV-formatted log string
    """
    kept_clusters, actions = automatic_size_filter(assigns, size_threshold)
    
    action_log = generate_action_log(actions, log_path)
    
    # Print summary
    n_total = len(np.unique(assigns[assigns != 0]))
    n_discarded = len(actions)
    n_kept = len(kept_clusters)
    
    print(f"Phase 0 Automatic Filtering:")
    print(f"  Total clusters: {n_total}")
    print(f"  Discarded (< {size_threshold} spikes): {n_discarded} ({n_discarded/n_total*100:.1f}%)")
    print(f"  Kept for Phase 1: {n_kept} ({n_kept/n_total*100:.1f}%)")
    
    if log_path:
        print(f"  Action log saved: {log_path}")
    
    return kept_clusters, action_log
