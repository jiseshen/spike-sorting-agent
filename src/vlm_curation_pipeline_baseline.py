"""
Baseline VLM-assisted curation pipeline.

VLM only judges neuronal validity (Phase 1: KEEP/DISCARD).
Split and merge decisions are made using heuristic correlation/ISI thresholds.

This serves as a baseline comparison to the pure VLM pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .cluster_manager import ClusterManager
from .cluster_features import compute_isi_violation_rate
from .auto_filter import automatic_size_filter
from .vlm_runner import (
    vlm_neuronal_validity_check,
    compute_waveform_correlation,
    compute_merged_isi_violation_rate,
)
from .agent_context import QUALITY_THRESHOLDS


class BaselineVLMCurationPipeline:
    """
    Baseline pipeline: VLM for neuronal validity, heuristics for split/merge.
    
    Workflow:
    - Phase 0: Automatic size filter (< 500 spikes)
    - Phase 1: VLM judges neuronal validity (KEEP/DISCARD only, no SPLIT)
              + Heuristic split by amplitude CV threshold
    - Phase 2: Heuristic merge by correlation + ISI thresholds
    - Phase 3: Final filter (< 5000 spikes after all merges)
    """
    
    def __init__(
        self,
        manager: ClusterManager,
        waveforms: np.ndarray,
        spike_times: np.ndarray,
        sampling_rate: float = 30000.0,
        provider: str = "gpt4o",
        model: str = "gpt-4o",
        use_mock: bool = False,
    ):
        self.manager = manager
        self.waveforms = waveforms
        self.spike_times = spike_times
        self.sampling_rate = sampling_rate
        self.provider = provider
        self.model = model
        self.use_mock = use_mock
        self.action_log = []
        
    def run(self, output_dir: Optional[Path] = None) -> ClusterManager:
        """Run full baseline pipeline."""
        print("=" * 80)
        print("BASELINE VLM-ASSISTED CURATION PIPELINE")
        print("=" * 80)
        
        # Phase 0: Automatic size filter
        print("\n[Phase 0] Automatic size filtering...")
        self.run_phase0_automatic_filtering()
        
        # Phase 1: VLM neuronal validity + heuristic split
        print("\n[Phase 1] VLM neuronal validity + heuristic split...")
        self.run_phase1_neuronal_validity()
        
        # Phase 2: Heuristic merge decisions
        print("\n[Phase 2] Heuristic merge decisions...")
        self.run_phase2_heuristic_merge()
        
        # Phase 3: Final size filter
        print("\n[Phase 3] Final size filter (< 5000 spikes)...")
        self.run_phase3_final_filter()
        
        # Export action log
        if output_dir:
            self._export_action_log(output_dir)
        
        print(f"\n[Pipeline Complete] {len(self.manager.get_active_clusters())} final clusters")
        return self.manager
    
    def run_phase0_automatic_filtering(self):
        """Phase 0: Discard clusters with < 500 spikes."""
        threshold = 500
        kept, discarded_actions = automatic_size_filter(
            self.manager.assigns,
            threshold,
        )
        
        for action in discarded_actions:
            cid = action.cluster_id
            self.manager.discard_cluster(cid)
            self.action_log.append({
                "phase": "Phase0_AutoFilter",
                "cluster_id": cid,
                "action": "DISCARD",
                "rationale": action.reasoning,
            })
        
        remaining = self.manager.get_active_clusters()
        print(f"  Discarded {len(discarded_actions)} clusters (< 500 spikes)")
        print(f"  Remaining: {len(remaining)} clusters")
    
    def run_phase1_neuronal_validity(self):
        """
        Phase 1: VLM judges neuronal validity + heuristic split.
        
        For each cluster:
        1. Call VLM with only KEEP/DISCARD options (no SPLIT)
        2. If VLM says DISCARD → discard
        3. If VLM says KEEP → iteratively split if ISI violation > 5%
        """
        clusters = self.manager.get_active_clusters()
        print(f"  Processing {len(clusters)} clusters...")
        
        clusters_to_process = list(clusters)  # Queue for iterative processing
        processed = set()
        
        while clusters_to_process:
            cid = clusters_to_process.pop(0)
            if cid in processed or cid not in self.manager.get_active_clusters():
                continue
            
            spike_indices = np.where(self.manager.assigns == cid)[0]
            if len(spike_indices) == 0:
                processed.add(cid)
                continue
            
            cluster_waveforms = self.waveforms[spike_indices]
            cluster_spike_times = self.spike_times[spike_indices]
            
            # Get cluster info
            info = self.manager.get_cluster_info(cid)
            unique_overclusters = info['overclusters'] if info else []
            
            # Compute ISI violation
            isi_violation, _, _ = compute_isi_violation_rate(cluster_spike_times)
            
            print(f"\n  Cluster {cid}: {len(spike_indices)} spikes, {len(unique_overclusters)} overclusters, ISI={isi_violation:.2%}")
            
            # Call VLM for neuronal validity ONLY (no split decision)
            decision = vlm_neuronal_validity_check(
                cluster_id=cid,
                waveforms=cluster_waveforms,
                spike_times=cluster_spike_times,
                sampling_rate=self.sampling_rate,
            )
            
            action = decision["action"]
            rationale = decision["rationale"]
            
            if action == "DISCARD":
                self.manager.discard_cluster(cid)
                self.action_log.append({
                    "phase": "Phase1_VLM_Neuronal",
                    "cluster_id": cid,
                    "action": "DISCARD",
                    "rationale": f"VLM: Not neuronal. {rationale}",
                })
                print(f"    → DISCARD (VLM: not neuronal)")
                processed.add(cid)
            
            elif action == "KEEP":  # VLM says neuronal, now check ISI
                if isi_violation > 0.005 and len(unique_overclusters) > 1:
                    # Heuristic split: ISI violation too high
                    print(f"    → KEEP (VLM: neuronal), but ISI={isi_violation:.2%} > 0.5% → heuristic split")
                    new_cluster_ids = self._heuristic_split_by_isi(cid, spike_indices, unique_overclusters, isi_violation)
                    # Add new clusters to processing queue
                    if new_cluster_ids:
                        clusters_to_process.extend(new_cluster_ids)
                    processed.add(cid)
                else:
                    # Keep as is
                    print(f"    → KEEP (VLM: neuronal, ISI={isi_violation:.2%})")
                    self.action_log.append({
                        "phase": "Phase1_VLM_Neuronal",
                        "cluster_id": cid,
                        "action": "KEEP",
                        "rationale": f"VLM: Neuronal. {rationale}. ISI={isi_violation:.2%}",
                    })
                    processed.add(cid)
    
    def _heuristic_split_by_isi(self, cluster_id: int, spike_indices: np.ndarray, 
                                 overcluster_ids: List[int], isi_violation: float) -> List[int]:
        """
        Heuristic split: Split by last merge in hierarchy tree.
        
        Strategy: Undo the most recent merge to reduce ISI violations.
        Returns list of new cluster IDs after split.
        """
        # Find the last merge for this cluster
        info = self.manager.get_cluster_info(cluster_id)
        if not info or 'last_merge_idx' not in info or info['last_merge_idx'] is None:
            print(f"      Cannot split: no merge history")
            return []
        
        last_merge_idx = info['last_merge_idx']
        
        # Get the two child overclusters from the last merge
        if last_merge_idx >= self.manager.hierarchy_tree.shape[1]:
            print(f"      Cannot split: invalid merge index")
            return []
        
        child1_oc = int(self.manager.hierarchy_tree[0, last_merge_idx])
        child2_oc = int(self.manager.hierarchy_tree[1, last_merge_idx])
        
        # Split into two groups
        split_groups = [[child1_oc], [child2_oc]]
            
        # Perform split
        new_cluster_ids = self.manager.split_by_overclusters(cluster_id, split_groups)
        
        self.action_log.append({
            "phase": "Phase1_Heuristic_Split",
            "cluster_id": cluster_id,
            "action": "SPLIT",
            "rationale": f"Heuristic split by ISI violation={isi_violation:.2%} > 5%. Undid last merge (OC {child1_oc} + {child2_oc}).",
        })
        
        print(f"      Split into clusters: {new_cluster_ids} (OC groups: {child1_oc}, {child2_oc})")
        
        # VLM validation of split results
        print(f"      Validating split results with VLM...")
        validated_ids = []
        for new_cid in new_cluster_ids:
            new_spike_indices = np.where(self.manager.assigns == new_cid)[0]
            if len(new_spike_indices) == 0:
                continue
            
            new_waveforms = self.waveforms[new_spike_indices]
            new_spike_times = self.spike_times[new_spike_indices]
            new_isi, _, _ = compute_isi_violation_rate(new_spike_times)
            
            # VLM neuronal validity check
            validation = vlm_neuronal_validity_check(
                cluster_id=new_cid,
                waveforms=new_waveforms,
                spike_times=new_spike_times,
                sampling_rate=self.sampling_rate,
            )
            
            if validation["action"] == "DISCARD":
                self.manager.discard_cluster(new_cid)
                self.action_log.append({
                    "phase": "Phase1_Split_Validation",
                    "cluster_id": new_cid,
                    "action": "DISCARD",
                    "rationale": f"VLM: Split result not neuronal. {validation['rationale']}",
                })
                print(f"        Cluster {new_cid}: DISCARD (not neuronal)")
            else:
                print(f"        Cluster {new_cid}: KEEP (neuronal, {len(new_spike_indices)} spikes, ISI={new_isi:.2%})")
                self.action_log.append({
                    "phase": "Phase1_Split_Validation",
                    "cluster_id": new_cid,
                    "action": "KEEP",
                    "rationale": f"VLM: Neuronal. {validation['rationale']}. ISI={new_isi:.2%}",
                })
                validated_ids.append(new_cid)
        
        return validated_ids
    
    def run_phase2_heuristic_merge(self):
        """
        Phase 2: Heuristic merge decisions.
        
        For each small cluster (< 4000 spikes):
        1. Compute correlation with all large clusters
        2. Find best match
        3. If correlation > 0.8 AND merged ISI < 5% → MERGE
        4. Otherwise → DISCARD
        """
        clusters = self.manager.get_active_clusters()
        threshold = 4000
        small_clusters = [cid for cid in clusters 
                         if np.sum(self.manager.assigns == cid) < threshold]
        large_clusters = [cid for cid in clusters 
                         if np.sum(self.manager.assigns == cid) >= threshold]
        
        print(f"  {len(small_clusters)} small clusters, {len(large_clusters)} large clusters")
        
        if len(large_clusters) == 0:
            print("  No large clusters to merge into. Discarding all small clusters.")
            for cid in small_clusters:
                self.manager.discard_cluster(cid)
                self.action_log.append({
                    "phase": "Phase2_Heuristic_Merge",
                    "cluster_id": cid,
                    "action": "DISCARD",
                    "rationale": "No large clusters available for merge",
                })
            return
        
        for small_cid in small_clusters:
            small_indices = np.where(self.manager.assigns == small_cid)[0]
            if len(small_indices) == 0:
                continue
            
            small_waveforms = self.waveforms[small_indices]
            small_spike_times = self.spike_times[small_indices]
            
            # Check if small cluster itself has high ISI violations
            small_isi, _, _ = compute_isi_violation_rate(small_spike_times)
            
            print(f"\n  Cluster {small_cid}: {len(small_indices)} spikes, ISI violation={small_isi:.2%}")
            
            # Early discard: if small cluster has high ISI and few spikes, discard directly
            if small_isi > 0.1 and len(small_indices) < 1000:
                self.manager.discard_cluster(small_cid)
                self.action_log.append({
                    "phase": "Phase2_Heuristic_Merge",
                    "cluster_id": small_cid,
                    "action": "DISCARD",
                    "rationale": f"High ISI violation ({small_isi:.2%}) with few spikes ({len(small_indices)})",
                })
                print(f"    → DISCARD (high ISI + few spikes)")
                continue
            
            # Find best match
            best_corr = -1
            best_large_cid = None
            best_merged_isi = 1.0
            
            for large_cid in large_clusters:
                large_indices = np.where(self.manager.assigns == large_cid)[0]
                large_waveforms = self.waveforms[large_indices]
                large_spike_times = self.spike_times[large_indices]
                
                corr = compute_waveform_correlation(small_waveforms, large_waveforms)
                merged_isi = compute_merged_isi_violation_rate(small_spike_times, large_spike_times)
                
                if corr > best_corr:
                    best_corr = corr
                    best_large_cid = large_cid
                    best_merged_isi = merged_isi
            
            print(f"    Best match: Cluster {best_large_cid}, corr={best_corr:.3f}, merged_isi={best_merged_isi:.2%}")
            
            # Decision (use 0.3 correlation threshold and 0.005 ISI threshold)
            if best_corr > 0.3 and best_merged_isi < 0.005:
                # MERGE
                self.manager.merge_clusters([small_cid, best_large_cid], target_id=best_large_cid)
                self.action_log.append({
                    "phase": "Phase2_Heuristic_Merge",
                    "cluster_id": small_cid,
                    "action": "MERGE",
                    "rationale": f"Heuristic merge into {best_large_cid}: corr={best_corr:.3f}, merged_isi={best_merged_isi:.2%}",
                })
                print(f"    → MERGE into {best_large_cid}")
            else:
                # DISCARD
                self.manager.discard_cluster(small_cid)
                self.action_log.append({
                    "phase": "Phase2_Heuristic_Merge",
                    "cluster_id": small_cid,
                    "action": "DISCARD",
                    "rationale": f"Cannot merge: corr={best_corr:.3f} (<0.3) or merged_isi={best_merged_isi:.2%} (>0.5%)",
                })
                print(f"    → DISCARD (cannot merge)")
    
    def run_phase3_final_filter(self):
        """
        Phase 3: Discard clusters with < 5000 spikes after all merges.
        
        This ensures final valid clusters meet the target size threshold.
        """
        clusters = self.manager.get_active_clusters()
        discarded = []
        
        threshold = 5000
        for cid in clusters:
            n_spikes = np.sum(self.manager.assigns == cid)
            if n_spikes < threshold:
                self.manager.discard_cluster(cid)
                discarded.append(cid)
                self.action_log.append({
                    "phase": "Phase3_Final_Filter",
                    "cluster_id": cid,
                    "action": "DISCARD",
                    "rationale": f"Final filter: {n_spikes} < {threshold} spikes",
                })
        
        print(f"  Discarded {len(discarded)} clusters (< {threshold} spikes)")
        remaining = self.manager.get_active_clusters()
        print(f"  Final: {len(remaining)} clusters")
    
    def _export_action_log(self, output_dir: Path):
        """Export action log to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.action_log)
        output_path = output_dir / "baseline_pipeline_actions.csv"
        df.to_csv(output_path, index=False)
        print(f"\n[Action Log] Saved to {output_path}")
