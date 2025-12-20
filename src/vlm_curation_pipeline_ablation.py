"""
ABLATION TEST: Pure VLM-driven curation pipeline WITHOUT quantitative metrics.

This is identical to vlm_curation_pipeline_pure.py except:
- Uses vlm_runner_ablation instead of vlm_runner
- VLM receives NO numerical metrics in prompts (no spike counts, ISI rates, correlations)
- VLM makes decisions based ONLY on visual information from images

Workflow:
- Phase 0: Auto size filter (<500 spikes only)
- Phase 1: Iterative VLM split/discard (until all clusters are clean)
- Phase 2: VLM merge/discard for small clusters (compare with each large cluster)
- Phase 3: Final size filter (discard <5000 spikes to ensure quality targets met)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pathlib import Path
import csv

from .cluster_manager import ClusterManager
from .cluster_features import ClusterFeatures
from .auto_filter import automatic_size_filter
from .vlm_runner_ablation import (
    vlm_phase1_cluster_decision,
    vlm_phase2_merge_decision,
)


class PureVLMCurationPipeline:
    """
    Pure VLM-driven curation pipeline.
    
    VLM judges all decisions based on neuronal criteria context.
    No rule-based heuristics for shape/ISI - all visual + context-driven.
    """
    
    def __init__(
        self,
        manager: ClusterManager,
        features: ClusterFeatures,
        sampling_rate: float = 30000.0,
        auto_discard_threshold: int = 500,
        small_cluster_threshold: int = 4000,
        final_minimum_threshold: int = 5000,
        provider: str = "gpt4o",
        model: str = "gpt-4o",
        use_mock: bool = False,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            manager: ClusterManager instance
            features: ClusterFeatures instance
            sampling_rate: Hz
            auto_discard_threshold: Minimum spike count for Phase 0 auto-discard
            small_cluster_threshold: Threshold for Phase 2 merge consideration (≥4000 = large)
            final_minimum_threshold: Final threshold after all merges (Phase 3)
            provider: VLM provider ("gpt4o", "claude")
            model: Model name (e.g., "gpt-4o", "o1-preview", "claude-3-5-sonnet-20241022")
            use_mock: If True, use mock VLM responses instead of real API
            temperature: Temperature for vision models (0.0-2.0), ignored for reasoning models
            reasoning_effort: Effort level for reasoning models ("low"/"medium"/"high"/None), ignored for vision
            output_dir: Directory to save VLM inputs (images + prompts)
        """
        self.manager = manager
        self.features = features
        self.sampling_rate = sampling_rate
        self.auto_discard_threshold = auto_discard_threshold
        self.small_cluster_threshold = small_cluster_threshold
        self.final_minimum_threshold = final_minimum_threshold
        self.provider = provider
        self.model = model
        self.use_mock = use_mock
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.output_dir = output_dir
        
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
    
    def get_overcluster_composition(self, cluster_id: int) -> List[int]:
        """Get list of overcluster IDs in this cluster."""
        info = self.manager.get_cluster_info(cluster_id)
        return info['overclusters'] if info else []
    
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
        
        Only rule-based step - tiny clusters are always noise.
        
        Returns:
            List of kept cluster IDs
        """
        print("\n" + "="*70)
        print("PHASE 0: AUTOMATIC SIZE FILTERING (<500 spikes)")
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
    
    def run_phase1_iterative_vlm_curation(
        self,
        cluster_ids: List[int],
    ) -> List[int]:
        """
        Phase 1: Iterative VLM-driven split/discard.
        
        For each cluster, VLM judges:
        1. Is it neuronal? (if not → DISCARD)
        2. Does it need split? (if yes → SPLIT and recurse on results)
        3. Is it clean? (if yes → KEEP)
        
        No iteration limit - VLM decides when splitting is complete.
        
        Args:
            cluster_ids: Clusters to curate
            
        Returns:
            List of final clean cluster IDs
        """
        print("\n" + "="*70)
        print("PHASE 1: ITERATIVE VLM CURATION (SPLIT/DISCARD)")
        print("="*70)
        
        final_clusters = []
        
        for cid in cluster_ids:
            print(f"\n→ Processing cluster {cid}...")
            clean_ids = self._iterative_cluster_curation(cid, iteration=1)
            final_clusters.extend(clean_ids)
        
        print(f"\n✓ Phase 1 complete: {len(final_clusters)} clean clusters")
        return final_clusters
    
    def _iterative_cluster_curation(
        self,
        cluster_id: int,
        iteration: int,
    ) -> List[int]:
        """
        Recursively curate a single cluster until VLM says it's clean.
        
        No iteration limit - continues until VLM returns KEEP or DISCARD.
        
        Returns:
            List of clean cluster IDs (may be empty if discarded, or multiple if split)
        """
        # Get cluster data
        waveforms = self.get_waveforms(cluster_id)
        spike_times = self.get_spike_times(cluster_id)
        overcluster_comp = self.get_overcluster_composition(cluster_id)
        
        if len(spike_times) == 0:
            print(f"  ✗ Cluster {cluster_id} has no spikes (already discarded)")
            return []
        
        print(f"  [Iteration {iteration}] VLM judging cluster {cluster_id} ({len(spike_times)} spikes)...")
        
        # Call VLM with hierarchy tree
        decision = vlm_phase1_cluster_decision(
            cluster_id=cluster_id,
            waveforms=waveforms,
            spike_times=spike_times,
            overcluster_composition=overcluster_comp,
            hierarchy_tree=self.manager.hierarchy_tree,
            sampling_rate=self.sampling_rate,
            provider=self.provider,
            model=self.model,
            use_mock=self.use_mock,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            output_dir=self.output_dir,
        )
        
        action = decision['action']
        rationale = decision['rationale']
        
        if action == "DISCARD":
            self.log_action(
                action="DISCARD",
                cluster_ids=[cluster_id],
                reason=f"VLM: {rationale}",
                phase=f"Phase1_Iter{iteration}",
            )
            self.manager.discard_cluster(cluster_id)
            return []
        
        elif action == "KEEP":
            print(f"  ✓ VLM: KEEP cluster {cluster_id} - {rationale}")
            self.log_action(
                action="KEEP",
                cluster_ids=[cluster_id],
                reason=f"VLM: {rationale}",
                phase=f"Phase1_Iter{iteration}",
            )
            return [cluster_id]
        
        elif action == "SPLIT":
            # We ignore split_groups; perform MATLAB-style last-merge split
            print(f"  → VLM: SPLIT cluster {cluster_id} (hierarchical last-merge) - {rationale}")
            self.log_action(
                action="SPLIT",
                cluster_ids=[cluster_id],
                reason=f"VLM: {rationale} | last-merge split",
                phase=f"Phase1_Iter{iteration}",
            )
            try:
                new_cluster_ids = self.manager.split_last_merge(cluster_id)
            except Exception as e:
                print(f"  ✗ Split failed for cluster {cluster_id}: {e}")
                return [cluster_id]
            
            # Recursively curate each split result
            all_clean = []
            for new_cid in new_cluster_ids:
                clean_ids = self._iterative_cluster_curation(new_cid, iteration + 1)
                all_clean.extend(clean_ids)
            
            return all_clean
        
        else:
            print(f"  ⚠ Unknown VLM action '{action}', discarding cluster {cluster_id}")
            self.manager.discard_cluster(cluster_id)
            return []
    
    def _apply_vlm_split(
        self,
        cluster_id: int,
        split_groups: List[List[int]],
    ) -> List[int]:
        """
        Apply VLM-recommended split by grouping overclusters.
        
        Args:
            cluster_id: Cluster to split
            split_groups: List of lists, each containing overcluster IDs
            
        Returns:
            List of new cluster IDs after split
        """
        # Build mapping: overcluster -> subgroup_index
        overcluster_to_group = {}
        for group_idx, overcluster_ids in enumerate(split_groups):
            for oc_id in overcluster_ids:
                overcluster_to_group[oc_id] = group_idx
        
        # Use ClusterManager's split_by_overclusters
        new_ids = self.manager.split_by_overclusters(cluster_id, overcluster_to_group)
        
        return new_ids
    
    def run_phase2_vlm_merge_decisions(
        self,
        all_cluster_ids: List[int],
    ) -> List[int]:
        """
        Phase 2: VLM merge/discard decisions for small clusters.
        
        Strategy:
        1. Partition clusters into small (<4000) and large (≥4000)
        2. For each large cluster: VLM checks if it's valid (not DISCARD)
        3. For each small cluster: VLM compares with each valid large cluster
           - If MERGE: merge small into large
           - If DISCARD: discard small cluster (invalid)
           - If NOT_MERGE: try next large cluster
        4. If no valid merge target found, discard small cluster
        
        Args:
            all_cluster_ids: All clusters after Phase 1
            
        Returns:
            List of final cluster IDs
        """
        print("\n" + "="*70)
        print("PHASE 2: VLM MERGE/DISCARD DECISIONS")
        print("="*70)
        
        # Partition into small and large
        small_ids = []
        large_ids = []
        
        for cid in all_cluster_ids:
            n_spikes = len(self.get_spike_times(cid))
            if n_spikes < self.small_cluster_threshold:
                small_ids.append(cid)
            else:
                large_ids.append(cid)
        
        print(f"Small clusters (<{self.small_cluster_threshold}): {len(small_ids)}")
        print(f"Large clusters (≥{self.small_cluster_threshold}): {len(large_ids)}")
        
        if not small_ids:
            print("✓ No small clusters to process")
            return list(all_cluster_ids)
        
        if not large_ids:
            print("⚠ No large clusters - discarding all small clusters")
            for cid in small_ids:
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[cid],
                    reason="No large clusters to merge into",
                    phase="Phase2",
                )
                self.manager.discard_cluster(cid)
            return []
        
        # Pre-filter large clusters: VLM checks each one
        # If VLM says DISCARD for a large cluster, skip it as merge target
        print(f"\n→ Pre-validating {len(large_ids)} large clusters...")
        valid_large_ids = []
        
        for large_cid in large_ids:
            large_waveforms = self.get_waveforms(large_cid)
            large_spike_times = self.get_spike_times(large_cid)
            overcluster_comp = self.get_overcluster_composition(large_cid)
            
            if len(large_spike_times) == 0:
                continue
            
            print(f"  🔍 VLM validating large cluster {large_cid} ({len(large_spike_times)} spikes)...")
            
            # Use Phase1 decision to check if cluster itself is valid
            decision = vlm_phase1_cluster_decision(
                cluster_id=large_cid,
                waveforms=large_waveforms,
                spike_times=large_spike_times,
                overcluster_composition=overcluster_comp,
                hierarchy_tree=self.manager.hierarchy_tree,
                sampling_rate=self.sampling_rate,
                provider=self.provider,
                model=self.model,
                use_mock=self.use_mock,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                output_dir=self.output_dir,
            )
            
            if decision['action'] == "DISCARD":
                print(f"  ✗ VLM: DISCARD large cluster {large_cid} - {decision['rationale']}")
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[large_cid],
                    reason=f"VLM (pre-validation): {decision['rationale']}",
                    phase="Phase2_PreValidation",
                )
                self.manager.discard_cluster(large_cid)
            else:
                print(f"  ✓ Large cluster {large_cid} validated (not DISCARD)")
                valid_large_ids.append(large_cid)
        
        print(f"\n✓ {len(valid_large_ids)}/{len(large_ids)} large clusters are valid merge targets")
        
        if not valid_large_ids:
            print("⚠ No valid large clusters - discarding all small clusters")
            for cid in small_ids:
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[cid],
                    reason="No valid large clusters to merge into",
                    phase="Phase2",
                )
                self.manager.discard_cluster(cid)
            return []
        
        # Process each small cluster
        final_ids = set(valid_large_ids)  # Keep all valid large clusters
        
        for small_cid in small_ids:
            print(f"\n→ Processing small cluster {small_cid}...")
            
            small_waveforms = self.get_waveforms(small_cid)
            small_spike_times = self.get_spike_times(small_cid)
            
            if len(small_spike_times) == 0:
                continue
            
            # Try merging with each large cluster (VLM decides)
            merged = False
            
            for large_cid in valid_large_ids:
                large_waveforms = self.get_waveforms(large_cid)
                large_spike_times = self.get_spike_times(large_cid)
                
                print(f"  🔍 VLM comparing {small_cid} with {large_cid}...")
                
                decision = vlm_phase2_merge_decision(
                    small_cluster_id=small_cid,
                    small_waveforms=small_waveforms,
                    small_spike_times=small_spike_times,
                    large_cluster_id=large_cid,
                    large_waveforms=large_waveforms,
                    large_spike_times=large_spike_times,
                    sampling_rate=self.sampling_rate,
                    provider=self.provider,
                    model=self.model,
                    use_mock=self.use_mock,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    output_dir=self.output_dir,
                )
                
                action = decision['action']
                
                if action == "MERGE":
                    print(f"  ✓ VLM: MERGE {small_cid} → {large_cid}")
                    self.log_action(
                        action="MERGE",
                        cluster_ids=[small_cid, large_cid],
                        reason=f"VLM: {decision['rationale']}",
                        phase="Phase2",
                    )
                    self.manager.merge_clusters([small_cid, large_cid], target_id=large_cid)
                    merged = True
                    break  # Merged, move to next small cluster
                
                elif action == "DISCARD":
                    # Small cluster itself is invalid (bad shape/too noisy)
                    print(f"  ✗ VLM: DISCARD {small_cid} (invalid cluster)")
                    self.log_action(
                        action="DISCARD",
                        cluster_ids=[small_cid],
                        reason=f"VLM: {decision['rationale']}",
                        phase="Phase2",
                    )
                    self.manager.discard_cluster(small_cid)
                    merged = True  # Stop trying other large clusters
                    break
                
                else:  # NOT_MERGE
                    print(f"  → VLM: NOT_MERGE with {large_cid} (different units) - {decision['rationale']}")
                    # Continue trying other large clusters
            
            if not merged:
                print(f"  → All large clusters returned NOT_MERGE, discarding {small_cid}")
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[small_cid],
                    reason="VLM: No compatible merge target (all returned NOT_MERGE)",
                    phase="Phase2",
                )
                self.manager.discard_cluster(small_cid)
        
        # Get final active clusters
        final_ids = self.manager.get_active_clusters()
        
        print(f"\n✓ Phase 2 complete: {len(final_ids)} clusters before final filter")
        return final_ids
    
    def run_phase3_final_filter(self, cluster_ids: List[int]) -> List[int]:
        """
        Phase 3: Final size filter - discard clusters < 5000 spikes.
        
        This ensures final valid clusters meet the target size threshold.
        Rationale: Target is ≥5000 spikes per valid cluster after all merges.
        
        Args:
            cluster_ids: Clusters after Phase 2
            
        Returns:
            List of final valid cluster IDs
        """
        print("\n" + "="*70)
        print(f"PHASE 3: FINAL SIZE FILTER (<{self.final_minimum_threshold} spikes)")
        print("="*70)
        
        discarded = []
        kept = []
        
        for cid in cluster_ids:
            spike_times = self.get_spike_times(cid)
            n_spikes = len(spike_times)
            
            if n_spikes < self.final_minimum_threshold:
                print(f"  ✗ Cluster {cid}: {n_spikes} spikes < {self.final_minimum_threshold} → DISCARD")
                self.log_action(
                    action="DISCARD",
                    cluster_ids=[cid],
                    reason=f"Final filter: {n_spikes} < {self.final_minimum_threshold} spikes",
                    phase="Phase3_FinalFilter",
                )
                self.manager.discard_cluster(cid)
                discarded.append(cid)
            else:
                print(f"  ✓ Cluster {cid}: {n_spikes} spikes ≥ {self.final_minimum_threshold} → KEEP")
                kept.append(cid)
        
        print(f"\n✓ Phase 3 complete: {len(discarded)} discarded, {len(kept)} final valid clusters")
        return kept
    
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
        Execute full pure VLM-driven curation pipeline.
        
        Returns:
            List of final cluster IDs
        """
        print("\n" + "="*70)
        print("PURE VLM-DRIVEN CURATION PIPELINE")
        print("="*70)
        
        # Phase 0: Auto size filter (only rule-based step)
        kept_ids = self.run_phase0_automatic_filtering()
        
        if not kept_ids:
            print("\n✓ All clusters filtered out in Phase 0")
            return []
        
        # Phase 1: Iterative VLM split/discard
        clean_ids = self.run_phase1_iterative_vlm_curation(kept_ids)
        
        if not clean_ids:
            print("\n✓ All clusters discarded in Phase 1")
            return []
        
        # Phase 2: VLM merge/discard for small clusters
        phase2_ids = self.run_phase2_vlm_merge_decisions(clean_ids)
        
        if not phase2_ids:
            print("\n✓ All clusters discarded in Phase 2")
            return []
        
        # Phase 3: Final size filter (≥5000 spikes)
        final_ids = self.run_phase3_final_filter(phase2_ids)
        
        print("\n" + "="*70)
        print(f"PIPELINE COMPLETE: {len(final_ids)} final valid clusters")
        print("="*70)
        
        return final_ids
