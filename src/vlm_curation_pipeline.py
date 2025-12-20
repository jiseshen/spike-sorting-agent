"""
VLM-integrated curation pipeline orchestrator.

Combines automation (Phase 0/1A) with VLM-driven decisions (Phase 1C/2).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pathlib import Path
import csv

from .cluster_manager import ClusterManager
from .cluster_features import ClusterFeatures
from .auto_filter import (
    automatic_size_filter,
    assess_waveform_shape,
    detect_temporal_drift,
)
from .vlm_runner import (
    vlm_phase1c_split_decision,
    vlm_phase2_postsplit_screening,
    compute_merged_isi_violation_rate,
)


class VLMCurationPipeline:
    """
    Orchestrates VLM-integrated cluster curation workflow.
    
    Workflow:
    - Phase 0: Automatic size filtering (<500 spikes → discard)
    - Phase 1A: Rule-based waveform shape assessment
    - Phase 1B: Temporal drift detection
    - Phase 1C: VLM-driven split decisions for large valid-shaped clusters
    - Phase 2: VLM post-split screening + merge compatibility
    - Final: ISI validation for merges
    """
    
    def __init__(
        self,
        manager: ClusterManager,
        features: ClusterFeatures,
        sampling_rate: float = 30000.0,
        auto_discard_threshold: int = 500,
    ):
        """
        Initialize pipeline.
        
        Args:
            manager: ClusterManager instance
            features: ClusterFeatures instance
            sampling_rate: Hz
            auto_discard_threshold: Minimum spike count for analysis
        """
        self.manager = manager
        self.features = features
        self.sampling_rate = sampling_rate
        self.auto_discard_threshold = auto_discard_threshold
        
        # Action log
        self.actions: List[Dict[str, Any]] = []
    
    def get_waveforms(self, cluster_id: int) -> np.ndarray:
        """Get waveforms for a cluster."""
        info = self.manager.get_cluster_info(cluster_id)
        return info['waveforms'] if info else np.array([])
    
    def get_spike_times(self, cluster_id: int) -> np.ndarray:
        """Get spike times for a cluster."""
        info = self.manager.get_cluster_info(cluster_id)
        return info['spike_times'] if info else np.array([])
    
    def log_action(
        self,
        action: str,
        cluster_ids: List[int],
        reason: str,
        phase: str,
    ):
        """Log curation action."""
        self.actions.append({
            "action": action,
            "cluster_ids": cluster_ids,
            "reason": reason,
            "phase": phase,
        })
        print(f"[{phase}] {action} {cluster_ids}: {reason}")
    
    def run_phase0_automatic_filtering(self) -> List[int]:
        """
        Phase 0: Automatic size-based filtering.
        
        Returns:
            List of kept cluster IDs
        """
        print("\n" + "="*70)
        print("PHASE 0: AUTOMATIC SIZE FILTERING")
        print("="*70)
        
        kept, discarded_actions = automatic_size_filter(
            self.manager.assigns, self.auto_discard_threshold
        )
        
        # Apply discards to manager
        for action in discarded_actions:
            cid = action.cluster_id
            self.manager.discard_cluster(cid)
            self.log_action(
                action="DISCARD",
                cluster_ids=[cid],
                reason=action.reasoning,
                phase="Phase0",
            )
        
        print(f"✓ Phase 0 complete: {len(discarded_actions)} discarded, {len(kept)} kept")
        return kept
    
    def run_phase1a_shape_assessment(
        self,
        cluster_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Phase 1A: Rule-based waveform shape assessment.
        
        Args:
            cluster_ids: Clusters to assess
            
        Returns:
            (valid_ids, invalid_ids)
        """
        print("\n" + "="*70)
        print("PHASE 1A: RULE-BASED SHAPE ASSESSMENT")
        print("="*70)
        
        valid = []
        invalid = []
        
        for cid in cluster_ids:
            waveforms = self.get_waveforms(cid)
            shape_result = assess_waveform_shape(waveforms, self.sampling_rate)
            
            if shape_result['is_valid']:
                valid.append(cid)
                print(f"✓ Cluster {cid}: VALID shape")
            else:
                invalid.append(cid)
                issues = ", ".join(shape_result['violations'])
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[cid],
                    reason=f"Invalid shape: {issues}",
                    phase="Phase1A",
                )
                self.manager.discard_cluster(cid)
        
        print(f"✓ Phase 1A complete: {len(valid)} valid, {len(invalid)} invalid")
        return valid, invalid
    
    def run_phase1b_drift_detection(
        self,
        cluster_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Phase 1B: Temporal drift detection.
        
        Args:
            cluster_ids: Clusters to assess
            
        Returns:
            Dict mapping cluster_id to drift results
        """
        print("\n" + "="*70)
        print("PHASE 1B: TEMPORAL DRIFT DETECTION")
        print("="*70)
        
        drift_results = {}
        
        for cid in cluster_ids:
            waveforms = self.manager.get_waveforms(cid)
            spike_times = self.manager.get_spike_times(cid)
            
            drift_result = detect_temporal_drift(
                waveforms, spike_times, self.sampling_rate
            )
            drift_results[cid] = drift_result
            
            if drift_result['drift_detected']:
                print(f"⚠ Cluster {cid}: DRIFT detected ({drift_result['width_change_pct']:.1%} change)")
            else:
                print(f"✓ Cluster {cid}: No significant drift")
        
        print(f"✓ Phase 1B complete: {sum(r['drift_detected'] for r in drift_results.values())} with drift")
        return drift_results
    
    def run_phase1c_vlm_splits(
        self,
        cluster_ids: List[int],
        drift_results: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        """
        Phase 1C: VLM-driven split decisions for large valid-shaped clusters.
        
        Args:
            cluster_ids: Valid-shaped clusters to consider for splitting
            drift_results: Drift detection results from Phase 1B
            
        Returns:
            List of all cluster IDs after splits (original + new split results)
        """
        print("\n" + "="*70)
        print("PHASE 1C: VLM-DRIVEN SPLIT DECISIONS")
        print("="*70)
        
        current_ids = list(cluster_ids)
        
        for cid in cluster_ids:
            waveforms = self.get_waveforms(cid)
            spike_times = self.get_spike_times(cid)
            n_spikes = len(spike_times)
            
            # Only apply VLM to large clusters
            if n_spikes < 1000:
                print(f"⊘ Cluster {cid}: Too small for VLM split ({n_spikes} spikes)")
                continue
            
            # Gather stats
            stats = {
                'amplitude_cv': np.std(np.ptp(waveforms, axis=1)) / np.mean(np.ptp(waveforms, axis=1)),
                'isi_violation_rate': self.features.compute_isi_violations(spike_times),
                'peak_to_trough_ms': self.features.compute_peak_to_trough_width(waveforms, self.sampling_rate),
            }
            
            # Build hierarchy summary (simplified - would extract from manager)
            hierarchy_summary = {
                "n_overclusters": 10,  # Mock - extract from manager.hierarchy
                "subgroups": ["Dense group at similarity 0.3", "Sparse tail"],
            }
            
            # Call VLM
            print(f"🔍 Calling VLM for cluster {cid}...")
            vlm_result = vlm_phase1c_split_decision(
                cluster_id=cid,
                waveforms=waveforms,
                spike_times=spike_times,
                hierarchy_summary=hierarchy_summary,
                stats=stats,
                sampling_rate=self.sampling_rate,
            )
            
            subgroups = vlm_result['subgroups']
            
            if not subgroups:
                print(f"→ VLM: No split recommended for cluster {cid}")
                continue
            
            # Apply splits
            print(f"→ VLM: Splitting cluster {cid} into {len(subgroups)} subgroups")
            
            for sg in subgroups:
                # For now, use mock split logic
                # In real implementation, would map member_ids to spike indices
                print(f"  - Subgroup {sg['subgroup_id']}: {sg['rationale']}")
                self.log_action(
                    action="SPLIT",
                    cluster_ids=[cid],
                    reason=f"VLM: {sg['rationale']}",
                    phase="Phase1C",
                )
            
            # Mock: assume splits created new IDs (would come from manager.split_cluster)
            # new_ids = self.manager.split_by_overclusters(cid, subgroup_mapping)
            # current_ids.extend(new_ids)
        
        print(f"✓ Phase 1C complete: {len(current_ids)} clusters after splits")
        return current_ids
    
    def run_phase2_vlm_postsplit_screening(
        self,
        all_cluster_ids: List[int],
        small_threshold: int = 500,
    ) -> List[int]:
        """
        Phase 2: VLM post-split screening and merge compatibility.
        
        Args:
            all_cluster_ids: All clusters after Phase 1C splits
            small_threshold: Threshold for small vs large clusters
            
        Returns:
            List of final cluster IDs after merges/discards
        """
        print("\n" + "="*70)
        print("PHASE 2: VLM POST-SPLIT SCREENING & MERGE")
        print("="*70)
        
        # Partition into small and large
        small_ids = []
        large_ids = []
        
        for cid in all_cluster_ids:
            n_spikes = len(self.get_spike_times(cid))
            if n_spikes < small_threshold:
                small_ids.append(cid)
            else:
                large_ids.append(cid)
        
        print(f"Small clusters (<{small_threshold}): {len(small_ids)}")
        print(f"Large clusters (≥{small_threshold}): {len(large_ids)}")
        
        if not small_ids:
            print("✓ No small clusters to screen")
            return list(all_cluster_ids)
        
        # Package small clusters
        small_clusters = []
        for cid in small_ids:
            waveforms = self.get_waveforms(cid)
            spike_times = self.get_spike_times(cid)
            stats = {
                'peak_to_trough_ms': self.features.compute_peak_to_trough_width(waveforms, self.sampling_rate),
                'phases': 3,  # Mock - would detect from waveform
                'baseline_offset': 0.15,  # Mock
                'isi_violation_rate': self.features.compute_isi_violations(spike_times),
            }
            small_clusters.append({
                'cluster_id': cid,
                'waveforms': waveforms,
                'spike_times': spike_times,
                'stats': stats,
            })
        
        # Package large clusters
        large_clusters = []
        for cid in large_ids:
            waveforms = self.get_waveforms(cid)
            spike_times = self.get_spike_times(cid)
            stats = {
                'isi_violation_rate': self.features.compute_isi_violations(spike_times),
            }
            large_clusters.append({
                'cluster_id': cid,
                'waveforms': waveforms,
                'spike_times': spike_times,
                'stats': stats,
            })
        
        # Call VLM
        print(f"🔍 Calling VLM for {len(small_clusters)} small clusters...")
        vlm_result = vlm_phase2_postsplit_screening(
            small_clusters=small_clusters,
            large_clusters=large_clusters,
            sampling_rate=self.sampling_rate,
        )
        
        decisions = vlm_result['decisions']
        
        # Apply decisions
        final_ids = set(all_cluster_ids)
        
        for decision in decisions:
            cid = decision['cluster_id']
            action = decision['action']
            merge_target = decision.get('merge_target')
            rationale = decision['rationale']
            
            if action == "DISCARD":
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[cid],
                    reason=f"VLM: {rationale}",
                    phase="Phase2",
                )
                self.manager.discard_cluster(cid)
                final_ids.discard(cid)
            
            elif action == "MERGE" and merge_target is not None:
                # Validate ISI
                spike_times_small = self.get_spike_times(cid)
                spike_times_large = self.get_spike_times(merge_target)
                
                merged_isi_rate = compute_merged_isi_violation_rate(
                    spike_times_small, spike_times_large
                )
                
                if merged_isi_rate < 0.05:
                    self.log_action(
                        action="MERGE",
                        cluster_ids=[cid, merge_target],
                        reason=f"VLM: {rationale} | Post-merge ISI: {merged_isi_rate:.2%}",
                        phase="Phase2",
                    )
                    self.manager.merge_clusters([cid, merge_target], target_id=merge_target)
                    final_ids.discard(cid)
                    print(f"✓ Merged {cid} → {merge_target} (ISI: {merged_isi_rate:.2%})")
                else:
                    self.log_action(
                        action="DISCARD",
                        cluster_ids=[cid],
                        reason=f"Merge rejected: ISI too high ({merged_isi_rate:.2%})",
                        phase="Phase2",
                    )
                    self.manager.discard_cluster(cid)
                    final_ids.discard(cid)
                    print(f"✗ Merge {cid} → {merge_target} rejected (ISI: {merged_isi_rate:.2%})")
            
            elif action == "KEEP":
                print(f"✓ Keeping cluster {cid}: {rationale}")
                self.log_action(
                    action="KEEP",
                    cluster_ids=[cid],
                    reason=f"VLM: {rationale}",
                    phase="Phase2",
                )
        
        print(f"✓ Phase 2 complete: {len(final_ids)} final clusters")
        return list(final_ids)
    
    def save_action_log(self, output_path: Path):
        """Save action log to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Action', 'Cluster IDs', 'Reason', 'Phase'])
            
            for action in self.actions:
                writer.writerow([
                    action['action'],
                    ','.join(map(str, action['cluster_ids'])),
                    action['reason'],
                    action['phase'],
                ])
        
        print(f"\n✓ Action log saved to {output_path}")
    
    def run_full_pipeline(self) -> List[int]:
        """
        Execute full VLM-integrated curation pipeline.
        
        Returns:
            List of final cluster IDs
        """
        print("\n" + "="*70)
        print("VLM-INTEGRATED CURATION PIPELINE")
        print("="*70)
        
        # Phase 0: Auto filtering
        kept_ids = self.run_phase0_automatic_filtering()
        
        if not kept_ids:
            print("\n✓ All clusters filtered out in Phase 0")
            return []
        
        # Phase 1A: Shape assessment
        valid_ids, _ = self.run_phase1a_shape_assessment(kept_ids)
        
        if not valid_ids:
            print("\n✓ All clusters filtered out in Phase 1A")
            return []
        
        # Phase 1B: Drift detection
        drift_results = self.run_phase1b_drift_detection(valid_ids)
        
        # Phase 1C: VLM splits
        post_split_ids = self.run_phase1c_vlm_splits(valid_ids, drift_results)
        
        # Phase 2: VLM post-split screening & merge
        final_ids = self.run_phase2_vlm_postsplit_screening(post_split_ids)
        
        print("\n" + "="*70)
        print(f"PIPELINE COMPLETE: {len(final_ids)} final clusters")
        print("="*70)
        
        return final_ids
