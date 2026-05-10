"""
Dynamic cluster management for split/merge operations.

Maintains cluster assignments and supports undo/redo operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class ClusterOperation:
    """Record of a single cluster operation for undo/redo."""
    op_type: str  # 'split', 'merge', 'discard'
    timestamp: int
    details: Dict
    old_assigns: np.ndarray
    new_assigns: np.ndarray


class ClusterManager:
    """
    Manages dynamic cluster assignments with split/merge operations.
    
    Features:
    - Split cluster by overcluster composition
    - Split cluster by custom spike mask
    - Merge multiple clusters
    - Discard (relabel to 0) clusters
    - Undo/redo operations
    - Track operation history
    """
    
    def __init__(self, 
                 initial_assigns: np.ndarray,
                 overcluster_assigns: np.ndarray,
                 hierarchy_tree: np.ndarray,
                 spike_times: np.ndarray,
                 waveforms: np.ndarray):
        """
        Initialize cluster manager.
        
        Args:
            initial_assigns: Starting cluster labels (e.g., hierarchy.assigns)
            overcluster_assigns: Overcluster labels for split reference
            hierarchy_tree: (4, n_merges) tree for split by similarity
            spike_times: (n_spikes,) spike times in seconds
            waveforms: (n_spikes, n_samples) waveforms
        """
        self.n_spikes = len(initial_assigns)
        self.overcluster_assigns = overcluster_assigns.copy()
        self.hierarchy_tree = hierarchy_tree
        self.spike_times = spike_times
        self.waveforms = waveforms
        
        # Current cluster assignments
        self.assigns = initial_assigns.copy()
        
        # Operation history
        self.history: List[ClusterOperation] = []
        self.history_index = -1  # Current position in history
        self._operation_counter = 0
        
        # Track which clusters have been modified
        self.modified_clusters: Set[int] = set()
        
        # Precompute overcluster composition for each hierarchy cluster
        self._build_cluster_composition()
    
    def _build_cluster_composition(self):
        """Build mapping: hierarchy_cluster -> list of overclusters."""
        self.cluster_composition: Dict[int, List[int]] = {}
        
        for cluster_id in np.unique(self.assigns):
            if cluster_id == 0:  # Skip noise
                continue
            mask = (self.assigns == cluster_id)
            overclusters = np.unique(self.overcluster_assigns[mask])
            self.cluster_composition[cluster_id] = overclusters.tolist()
    
    def _record_operation(self, op_type: str, details: Dict, 
                         old_assigns: np.ndarray, new_assigns: np.ndarray):
        """Record operation for undo/redo."""
        # Remove any operations after current position (if we undid and then did new op)
        self.history = self.history[:self.history_index + 1]
        
        op = ClusterOperation(
            op_type=op_type,
            timestamp=self._operation_counter,
            details=details,
            old_assigns=old_assigns.copy(),
            new_assigns=new_assigns.copy()
        )
        self.history.append(op)
        self.history_index += 1
        self._operation_counter += 1
    
    def get_cluster_info(self, cluster_id: int) -> Dict:
        """Get information about a cluster."""
        mask = (self.assigns == cluster_id)
        n_spikes = mask.sum()
        
        if n_spikes == 0:
            return None
        
        # Get overcluster composition
        overclusters = np.unique(self.overcluster_assigns[mask])
        
        # Get spike indices
        spike_indices = np.where(mask)[0]
        
        return {
            'cluster_id': cluster_id,
            'n_spikes': n_spikes,
            'overclusters': overclusters.tolist(),
            'n_overclusters': len(overclusters),
            'spike_indices': spike_indices,
            'spike_times': self.spike_times[mask],
            'waveforms': self.waveforms[mask],
            'modified': cluster_id in self.modified_clusters
        }
    
    def get_active_clusters(self) -> List[int]:
        """Get list of active (non-zero) cluster IDs."""
        unique = np.unique(self.assigns)
        return [int(c) for c in unique if c != 0]
    
    def split_by_overclusters(self, cluster_id: int, 
                             overcluster_groups: List[List[int]],
                             new_ids: Optional[List[int]] = None) -> List[int]:
        """
        Split cluster by grouping its constituent overclusters.
        
        Args:
            cluster_id: Cluster to split
            overcluster_groups: List of overcluster groups, e.g.,
                [[5, 17], [9, 18]] means split into 2 groups
            new_ids: Optional new cluster IDs for each group.
                If None, will auto-assign from max_id + 1
        
        Returns:
            List of new cluster IDs created
            
        Example:
            # Split cluster 1 by reverting low-similarity merges
            manager.split_by_overclusters(1, [[5, 17], [9, 18]])
        """
        old_assigns = self.assigns.copy()
        
        # Validate cluster exists
        mask = (self.assigns == cluster_id)
        if mask.sum() == 0:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        # Auto-assign new IDs if not provided
        if new_ids is None:
            max_id = int(self.assigns.max())
            new_ids = list(range(max_id + 1, max_id + 1 + len(overcluster_groups)))
        
        if len(new_ids) != len(overcluster_groups):
            raise ValueError(f"Need {len(overcluster_groups)} new IDs, got {len(new_ids)}")
        
        # Apply split
        for new_id, overcluster_group in zip(new_ids, overcluster_groups):
            for oc in overcluster_group:
                # Find spikes: belong to cluster_id AND overcluster oc
                split_mask = (self.assigns == cluster_id) & (self.overcluster_assigns == oc)
                self.assigns[split_mask] = new_id
        
        # Update hierarchy_tree: remove merge entries involving split clusters
        # MATLAB logic: find last merge entry where tree(:,1) == cluster_id, delete it
        tree_mask = (self.hierarchy_tree[0, :] == cluster_id)
        if tree_mask.any():
            # Find last (most recent) merge
            last_merge_idx = np.where(tree_mask)[0][-1]
            # Remove this merge from tree
            self.hierarchy_tree = np.delete(self.hierarchy_tree, last_merge_idx, axis=1)
        
        # Record operation
        self._record_operation(
            op_type='split',
            details={
                'original_cluster': cluster_id,
                'new_clusters': new_ids,
                'overcluster_groups': overcluster_groups,
                'method': 'overcluster'
            },
            old_assigns=old_assigns,
            new_assigns=self.assigns
        )
        
        # Mark as modified
        self.modified_clusters.add(cluster_id)
        self.modified_clusters.update(new_ids)
        
        # Update composition
        self._build_cluster_composition()
        
        return new_ids
    
    def split_by_mask(self, cluster_id: int, 
                     masks: List[np.ndarray],
                     new_ids: Optional[List[int]] = None) -> List[int]:
        """
        Split cluster by custom spike masks (e.g., from GMM, amplitude threshold).
        
        Args:
            cluster_id: Cluster to split
            masks: List of boolean masks for each new cluster
            new_ids: Optional new cluster IDs
        
        Returns:
            List of new cluster IDs
            
        Example:
            # Split by amplitude bimodality
            low_amp_mask = (amplitudes < threshold)
            high_amp_mask = (amplitudes >= threshold)
            manager.split_by_mask(1, [low_amp_mask, high_amp_mask])
        """
        old_assigns = self.assigns.copy()
        
        # Validate
        mask = (self.assigns == cluster_id)
        n_spikes = mask.sum()
        if n_spikes == 0:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        # Auto-assign IDs
        if new_ids is None:
            max_id = int(self.assigns.max())
            new_ids = list(range(max_id + 1, max_id + 1 + len(masks)))
        
        # Apply split
        for new_id, split_mask in zip(new_ids, masks):
            # Only reassign spikes that belong to cluster_id
            combined_mask = mask & split_mask
            self.assigns[combined_mask] = new_id
        
        # Record operation
        self._record_operation(
            op_type='split',
            details={
                'original_cluster': cluster_id,
                'new_clusters': new_ids,
                'n_spikes_per_group': [m.sum() for m in masks],
                'method': 'mask'
            },
            old_assigns=old_assigns,
            new_assigns=self.assigns
        )
        
        self.modified_clusters.add(cluster_id)
        self.modified_clusters.update(new_ids)
        self._build_cluster_composition()
        
        return new_ids

    def split_by_current_overclusters(self, cluster_id: int) -> List[int]:
        """
        Safe split fallback when hierarchy tree history is unavailable.

        Strategy:
        - Keep the largest overcluster under the original `cluster_id`
        - Split each remaining overcluster into its own new cluster

        This is intended as a robust fallback for simulated pipelines where
        hierarchy trees may be empty or incomplete.
        """
        mask = self.assigns == cluster_id
        if not np.any(mask):
            raise ValueError(f"Cluster {cluster_id} not found")

        ocs = self.overcluster_assigns[mask]
        uniq, counts = np.unique(ocs, return_counts=True)
        if len(uniq) <= 1:
            return [cluster_id]

        order = np.argsort(-counts)  # descending by size
        sorted_ocs = [int(uniq[i]) for i in order]
        keep_oc = sorted_ocs[0]
        split_ocs = sorted_ocs[1:]

        groups: List[List[int]] = [[keep_oc]] + [[oc] for oc in split_ocs]
        max_id = int(self.assigns.max())
        new_ids = [cluster_id] + list(range(max_id + 1, max_id + 1 + len(split_ocs)))

        return self.split_by_overclusters(cluster_id, groups, new_ids)
    
    def split_by_similarity_threshold(self, cluster_id: int, 
                                     similarity_threshold: float = 0.3) -> List[int]:
        """
        Automatic split by reverting merges below similarity threshold.
        
        Uses hierarchy tree to identify which overclusters were merged
        with low similarity, then splits them out.
        
        Args:
            cluster_id: Cluster to split
            similarity_threshold: Merges below this similarity are reverted
        
        Returns:
            List of new cluster IDs (empty if no split needed)
        """
        # Get overclusters in this cluster
        info = self.get_cluster_info(cluster_id)
        if info is None:
            return []
        
        overclusters = set(info['overclusters'])
        
        # Find merges involving these overclusters with low similarity
        low_sim_groups = []
        remaining = overclusters.copy()
        
        for i in range(self.hierarchy_tree.shape[1]):
            src1, src2, sim, _ = self.hierarchy_tree[:, i]
            src1, src2 = int(src1), int(src2)
            
            if src1 in overclusters or src2 in overclusters:
                if sim < similarity_threshold:
                    # Low similarity merge - split it out
                    group = {src2}  # The merged-in overcluster
                    low_sim_groups.append(list(group))
                    remaining.discard(src2)
        
        if len(low_sim_groups) == 0:
            return []  # No low-similarity merges to revert
        
        # First group is the remaining overclusters (keep original ID)
        groups = [list(remaining)] + low_sim_groups
        new_ids = [cluster_id] + [None] * len(low_sim_groups)
        
        return self.split_by_overclusters(cluster_id, groups, new_ids)
    
    def split_last_merge(self, cluster_id: int) -> List[int]:
        """
        Split cluster by undoing the last merge (MATLAB-style split).
        
        Finds the most recent merge in hierarchy_tree involving cluster_id,
        identifies which cluster was merged in, traces back all overclusters
        that belong to the merged-in cluster, and splits them out.
        
        This matches MATLAB split_cluster.m logic exactly.
        
        Args:
            cluster_id: Cluster to split
            
        Returns:
            List of [original_cluster_id, breakaway_cluster_id]
        """
        # Find last merge where tree[0, :] == cluster_id
        tree_mask = (self.hierarchy_tree[0, :] == cluster_id)
        if not tree_mask.any():
            raise ValueError(f"Cluster {cluster_id} has no merge history in tree")
        
        # Get last (most recent) merge
        last_merge_idx = np.where(tree_mask)[0][-1]
        breakaway = int(self.hierarchy_tree[1, last_merge_idx])  # The cluster that was merged in
        
        # Recursively find all overclusters that contribute to breakaway
        # Walk backwards through tree to find all merges that formed breakaway
        breakaway_overclusters = [breakaway]
        for i in range(last_merge_idx - 1, -1, -1):
            src1 = int(self.hierarchy_tree[0, i])
            src2 = int(self.hierarchy_tree[1, i])
            # If src1 is already in breakaway list, add src2
            if src1 in breakaway_overclusters:
                breakaway_overclusters.append(src2)
        
        # Get remaining overclusters (those that stay in original cluster)
        info = self.get_cluster_info(cluster_id)
        all_overclusters = set(info['overclusters'])
        breakaway_set = set(breakaway_overclusters)
        remaining_set = all_overclusters - breakaway_set
        
        # Split: remaining stays as cluster_id, breakaway gets new ID (or use breakaway[0])
        # MATLAB uses breakaway(1) as the new label
        groups = [list(remaining_set), list(breakaway_set)]
        new_ids = [cluster_id, breakaway]  # Keep original logic: use breakaway as new ID
        
        return self.split_by_overclusters(cluster_id, groups, new_ids)
    
    def merge_clusters(self, cluster_ids: List[int], 
                      target_id: Optional[int] = None) -> int:
        """
        Merge multiple clusters into one.
        
        Args:
            cluster_ids: List of cluster IDs to merge
            target_id: Target cluster ID (default: keep smallest ID)
        
        Returns:
            Final cluster ID
            
        Example:
            manager.merge_clusters([222, 218, 226])
        """
        old_assigns = self.assigns.copy()
        
        # Validate
        for cid in cluster_ids:
            if (self.assigns == cid).sum() == 0:
                raise ValueError(f"Cluster {cid} not found")
        
        # Use smallest ID as target if not specified
        if target_id is None:
            target_id = min(cluster_ids)
        
        # Merge all into target
        for cid in cluster_ids:
            if cid != target_id:
                self.assigns[self.assigns == cid] = target_id
        
        # Update hierarchy_tree: add merge entries
        # MATLAB logic: append [to from cs score] to tree
        # For simplicity, use connection_strength=1.0 and score=0 (would need ISI calculation for real score)
        for cid in cluster_ids:
            if cid != target_id:
                new_merge = np.array([[target_id], [cid], [1.0], [0.0]])
                self.hierarchy_tree = np.hstack([self.hierarchy_tree, new_merge])
        
        # Record operation
        self._record_operation(
            op_type='merge',
            details={
                'source_clusters': cluster_ids,
                'target_cluster': target_id,
                'n_spikes': (self.assigns == target_id).sum()
            },
            old_assigns=old_assigns,
            new_assigns=self.assigns
        )
        
        self.modified_clusters.update(cluster_ids)
        self._build_cluster_composition()
        
        return target_id
    
    def discard_cluster(self, cluster_id: int):
        """
        Discard cluster by merging to 0 (noise).
        
        This is equivalent to merge_clusters([cluster_id], target_id=0).
        
        Args:
            cluster_id: Cluster to discard
        """
        old_assigns = self.assigns.copy()
        
        mask = (self.assigns == cluster_id)
        n_spikes = mask.sum()
        if n_spikes == 0:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        # Merge to cluster 0 (noise)
        self.assigns[mask] = 0
        
        # Update hierarchy_tree: add merge entry [0, cluster_id, 0.0, 0.0]
        # Connection strength and score are 0 for discards
        new_merge = np.array([[0], [cluster_id], [0.0], [0.0]])
        self.hierarchy_tree = np.hstack([self.hierarchy_tree, new_merge])
        
        # Record operation
        self._record_operation(
            op_type='discard',
            details={
                'cluster_id': cluster_id,
                'n_spikes': n_spikes
            },
            old_assigns=old_assigns,
            new_assigns=self.assigns
        )
        
        self.modified_clusters.add(cluster_id)
        self._build_cluster_composition()
    
    def undo(self) -> bool:
        """
        Undo last operation.
        
        Returns:
            True if undo successful, False if at start of history
        """
        if self.history_index < 0:
            return False
        
        op = self.history[self.history_index]
        self.assigns = op.old_assigns.copy()
        self.history_index -= 1
        
        self._build_cluster_composition()
        return True
    
    def redo(self) -> bool:
        """
        Redo previously undone operation.
        
        Returns:
            True if redo successful, False if at end of history
        """
        if self.history_index >= len(self.history) - 1:
            return False
        
        self.history_index += 1
        op = self.history[self.history_index]
        self.assigns = op.new_assigns.copy()
        
        self._build_cluster_composition()
        return True
    
    def get_history(self) -> List[Dict]:
        """Get operation history as list of dicts."""
        return [
            {
                'index': i,
                'type': op.op_type,
                'timestamp': op.timestamp,
                'details': op.details,
                'active': i <= self.history_index
            }
            for i, op in enumerate(self.history)
        ]
    
    def export_sorting(self):
        """
        Export current assignments as SpikeInterface NumpySorting.
        
        Returns:
            NumpySorting object with current cluster assignments
        """
        from spikeinterface.core import NumpySorting
        
        # Get sampling frequency (assume 30kHz)
        fs = 30000.0
        
        # Convert spike times to samples
        spike_samples = np.round(self.spike_times * fs).astype(np.int64)
        
        # Get active clusters
        active_clusters = self.get_active_clusters()
        
        # Build spike trains
        spike_trains = {}
        for cid in active_clusters:
            mask = (self.assigns == cid)
            spike_trains[cid] = spike_samples[mask]
        
        return NumpySorting.from_times_labels(
            times_list=[spike_trains[cid] for cid in sorted(spike_trains.keys())],
            labels_list=list(sorted(spike_trains.keys())),
            sampling_frequency=fs
        )
    
    def get_statistics(self) -> Dict:
        """Get current clustering statistics."""
        active = self.get_active_clusters()
        
        stats = {
            'n_clusters': len(active),
            'n_spikes_assigned': (self.assigns != 0).sum(),
            'n_spikes_noise': (self.assigns == 0).sum(),
            'n_operations': len(self.history),
            'n_modified_clusters': len(self.modified_clusters),
            'cluster_sizes': {
                int(cid): (self.assigns == cid).sum() 
                for cid in active
            }
        }
        
        return stats
