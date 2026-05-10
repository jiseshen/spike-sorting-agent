"""
StudentRunner: wraps existing VLM decision functions for use inside TrajectoryRunner.

Calls vlm_phase1_cluster_decision / vlm_phase2_merge_decision from
src.agent.runner, returning structured responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.runner import vlm_phase1_cluster_decision, vlm_phase2_merge_decision
from src.cluster.manager import ClusterManager


class StudentRunner:
    """
    Calls the student VLM for a single curation decision.

    Args:
        student_model: Model name (e.g. "gpt-4o", "gpt-4.1").
        provider: VLM provider (e.g. "gpt4o", "claude").
        use_mock: If True, return placeholder responses.
        temperature: Sampling temperature.
        reasoning_effort: Optional reasoning effort flag.
    """

    def __init__(
        self,
        student_model: str,
        provider: str = "gpt4o",
        use_mock: bool = False,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.student_model = student_model
        self.provider = provider
        self.use_mock = use_mock
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def decide_phase1(
        self,
        cluster_id: int,
        manager: ClusterManager,
        sampling_rate: float,
        output_dir: Optional[Path] = None,
        step: int = 0,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, object]:
        """
        Phase 1 decision: KEEP / DISCARD / SPLIT for one cluster.

        Returns dict with keys: action, rationale.
        """
        info = manager.get_cluster_info(cluster_id)
        if info is None:
            return {"action": "DISCARD", "rationale": "Cluster not found."}

        out = None
        if output_dir is not None:
            out = Path(output_dir) / f"step{step:04d}_phase1"

        return vlm_phase1_cluster_decision(
            cluster_id=cluster_id,
            waveforms=info["waveforms"],
            spike_times=info["spike_times"],
            overcluster_composition=info["overclusters"],
            hierarchy_tree=manager.hierarchy_tree,
            sampling_rate=sampling_rate,
            provider=self.provider,
            model=self.student_model,
            use_mock=self.use_mock,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            output_dir=out,
            retrieved_examples=retrieved_examples,
        )

    def decide_phase2(
        self,
        small_cluster_id: int,
        large_cluster_id: int,
        manager: ClusterManager,
        sampling_rate: float,
        output_dir: Optional[Path] = None,
        step: int = 0,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, object]:
        """
        Phase 2 decision: MERGE / NOT_MERGE / DISCARD for a (small, large) cluster pair.

        Returns dict with keys: action, rationale.
        """
        small_info = manager.get_cluster_info(small_cluster_id)
        large_info = manager.get_cluster_info(large_cluster_id)
        if small_info is None or large_info is None:
            return {"action": "NOT_MERGE", "rationale": "One or both clusters not found."}

        out = None
        if output_dir is not None:
            out = Path(output_dir) / f"step{step:04d}_phase2"

        return vlm_phase2_merge_decision(
            small_cluster_id=small_cluster_id,
            small_waveforms=small_info["waveforms"],
            small_spike_times=small_info["spike_times"],
            large_cluster_id=large_cluster_id,
            large_waveforms=large_info["waveforms"],
            large_spike_times=large_info["spike_times"],
            sampling_rate=sampling_rate,
            provider=self.provider,
            model=self.student_model,
            use_mock=self.use_mock,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            output_dir=out,
            retrieved_examples=retrieved_examples,
        )
