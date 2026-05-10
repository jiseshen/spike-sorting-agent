"""
TrajectoryRunner: orchestrates the teacher-student interaction for one channel.

Workflow per channel:
  1. Load raw arrays (waveforms, spike_times, assigns, gt_assigns)
  2. Phase 0: auto-discard tiny clusters
  3. For each active cluster (phase 1 order):
       a. Student decides KEEP / DISCARD / SPLIT
       b. Oracle provides GT action + reasoning
       c. Teacher provides feedback given GT
       d. Record TrajectoryStep
       e. Apply GT action (teacher-forced) to advance state
  4. Repeat for phase 2 merge candidates
  5. Save trajectory.jsonl + summary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.cluster.manager import ClusterManager
from src.cluster.auto_filter import automatic_size_filter
from src.cluster.features import ClusterFeatures
from src.actions.oracle import best_oracle_action_for_cluster
from src.simulate.setting import SettingConfig
from src.agent.rag_memory import (
    ContinualRAGMemory,
    build_phase1_memory_entry,
    build_phase2_memory_entry,
)

from .record import TrajectoryStep, save_trajectory, trajectory_summary
from .student import StudentRunner
from .teacher import GroundTruthTeacher


class TrajectoryRunner:
    """
    Generates teacher-student interaction trajectories for one channel.

    Args:
        cfg: SettingConfig (contains teacher_criteria, teacher_style).
        student: Configured StudentRunner.
        teacher: Configured GroundTruthTeacher.
        auto_discard_threshold: Phase 0 spike count floor.
        small_cluster_threshold: Threshold for phase 2 merge candidacy.
        force: Re-run even if trajectory.jsonl exists.
        enable_rag_baseline: Enable continual RAG memory retrieval/insertion.
        rag_top_k: Number of retrieved examples used for few-shot context.
        rag_waveform_weight: Similarity weight for waveform template cosine.
        rag_feature_weight: Similarity weight for feature-vector cosine.
        rag_memory_path: Optional JSONL path for append-only memory persistence.
    """

    def __init__(
        self,
        cfg: SettingConfig,
        student: StudentRunner,
        teacher: GroundTruthTeacher,
        auto_discard_threshold: int = 500,
        small_cluster_threshold: int = 4000,
        force: bool = False,
        enable_rag_baseline: bool = False,
        rag_top_k: int = 3,
        rag_waveform_weight: float = 0.7,
        rag_feature_weight: float = 0.3,
        rag_memory_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg = cfg
        self.student = student
        self.teacher = teacher
        self.auto_discard_threshold = auto_discard_threshold
        self.small_cluster_threshold = small_cluster_threshold
        self.force = force
        self.enable_rag_baseline = enable_rag_baseline
        self.rag_top_k = rag_top_k
        self.rag_memory = (
            ContinualRAGMemory(
                memory_path=rag_memory_path,
                waveform_weight=rag_waveform_weight,
                feature_weight=rag_feature_weight,
                default_top_k=rag_top_k,
            )
            if enable_rag_baseline
            else None
        )

    def run(
        self,
        raw_dir: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> list[TrajectoryStep]:
        """
        Run the full trajectory for one channel.

        Args:
            raw_dir: Path to channel raw/ directory.
            output_dir: Where to write trajectory.jsonl (defaults to raw_dir/../trajectory/).

        Returns:
            List of TrajectoryStep objects.
        """
        raw_dir = Path(raw_dir)
        channel_id = raw_dir.parent.name
        setting_id = raw_dir.parent.parent.name

        if output_dir is None:
            output_dir = raw_dir.parent / "trajectory"
        output_dir = Path(output_dir)

        out_file = output_dir / "trajectory.jsonl"
        if out_file.exists() and not self.force:
            print(f"  [skip] {out_file} exists (use --force to overwrite)")
            from .record import load_trajectory
            return load_trajectory(out_file)

        # --- Load arrays ---
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

        # Phase 0: auto-discard
        _, filter_actions = automatic_size_filter(manager.assigns, self.auto_discard_threshold)
        for fa in filter_actions:
            manager.discard_cluster(fa.cluster_id)

        steps: list[TrajectoryStep] = []
        step_idx = 0

        # --- Phase 1: KEEP / DISCARD / SPLIT ---
        phase1_clusters = list(manager.get_active_clusters())
        for cid in phase1_clusters:
            if cid not in manager.get_active_clusters():
                continue
            info = manager.get_cluster_info(cid)
            if info is None:
                continue
            current_step = step_idx

            retrieved_examples: List[Dict[str, Any]] = []
            if self.rag_memory is not None:
                retrieved_examples = self.rag_memory.retrieve_phase1(
                    waveforms=info["waveforms"],
                    spike_times=info["spike_times"],
                    n_spikes=int(info["n_spikes"]),
                    top_k=self.rag_top_k,
                )

            student_resp = self.student.decide_phase1(
                cluster_id=cid,
                manager=manager,
                sampling_rate=Fs,
                output_dir=output_dir / "student_inputs",
                step=current_step,
                retrieved_examples=retrieved_examples,
            )

            oracle = best_oracle_action_for_cluster(
                cluster_id=cid,
                assigns=manager.assigns,
                gt_assigns=gt_assigns,
                active_clusters=manager.get_active_clusters(),
                purity_threshold=0.80,
            )

            context_summary = {
                "cluster_id": cid,
                "n_spikes": int(info["n_spikes"]) if info else 0,
                "n_overclusters": len(info["overclusters"]) if info else 0,
            }

            teacher_feedback = self.teacher.get_feedback(
                channel_id=channel_id,
                step=current_step,
                action_type=oracle.action_type.lower(),
                context_summary=context_summary,
                student_action=student_resp.get("action", ""),
                student_rationale=student_resp.get("rationale", ""),
                gt_action=oracle.action_type,
                gt_reasoning=oracle.gt_reasoning,
                images=[],   # images generated inside student; teacher uses same context
                output_dir=output_dir / "teacher_outputs",
            )

            step = TrajectoryStep(
                step=current_step,
                channel_id=channel_id,
                setting_id=setting_id,
                cluster_id=cid,
                action_phase="phase1",
                target_id=None,
                gt_action=oracle.action_type,
                gt_reasoning=oracle.gt_reasoning,
                student_action=student_resp.get("action", ""),
                student_rationale=student_resp.get("rationale", ""),
                student_model=self.student.student_model,
                teacher_feedback=teacher_feedback,
                teacher_model=self.teacher.teacher_model,
                n_spikes=int(info["n_spikes"]) if info else None,
                n_active_clusters=len(manager.get_active_clusters()),
            )
            steps.append(step)

            if self.rag_memory is not None:
                rag_entry = build_phase1_memory_entry(
                    channel_id=channel_id,
                    step=current_step,
                    cluster_id=cid,
                    waveforms=info["waveforms"],
                    spike_times=info["spike_times"],
                    gt_action=oracle.action_type,
                    gt_reasoning=oracle.gt_reasoning,
                    prompt_text=str(student_resp.get("prompt_text", "")),
                    image_paths=[str(p) for p in student_resp.get("image_paths", [])],
                )
                self.rag_memory.add(rag_entry)

            step_idx += 1

            # Apply GT action (teacher-forced) to advance state
            _apply_oracle(manager, oracle)

        # --- Phase 2: MERGE candidates ---
        active = manager.get_active_clusters()
        large_clusters = [c for c in active if manager.get_cluster_info(c)["n_spikes"] >= self.small_cluster_threshold]
        small_clusters = [c for c in active if manager.get_cluster_info(c)["n_spikes"] < self.small_cluster_threshold]

        for small_cid in small_clusters:
            if small_cid not in manager.get_active_clusters():
                continue
            for large_cid in large_clusters:
                if large_cid not in manager.get_active_clusters():
                    continue
                small_info = manager.get_cluster_info(small_cid)
                large_info = manager.get_cluster_info(large_cid)
                if small_info is None or large_info is None:
                    continue
                current_step = step_idx

                retrieved_examples: List[Dict[str, Any]] = []
                if self.rag_memory is not None:
                    retrieved_examples = self.rag_memory.retrieve_phase2(
                        small_waveforms=small_info["waveforms"],
                        small_spike_times=small_info["spike_times"],
                        large_waveforms=large_info["waveforms"],
                        large_spike_times=large_info["spike_times"],
                        top_k=self.rag_top_k,
                    )

                student_resp = self.student.decide_phase2(
                    small_cluster_id=small_cid,
                    large_cluster_id=large_cid,
                    manager=manager,
                    sampling_rate=Fs,
                    output_dir=output_dir / "student_inputs",
                    step=current_step,
                    retrieved_examples=retrieved_examples,
                )

                oracle = best_oracle_action_for_cluster(
                    cluster_id=small_cid,
                    assigns=manager.assigns,
                    gt_assigns=gt_assigns,
                    active_clusters=manager.get_active_clusters(),
                    purity_threshold=0.80,
                )
                # For phase 2 the oracle only tells MERGE or KEEP/DISCARD
                if oracle.action_type == "MERGE" and oracle.target_id != large_cid:
                    oracle_action_type = "NOT_MERGE"
                    oracle_reasoning = f"GT suggests merging {small_cid} with {oracle.target_id}, not {large_cid}."
                else:
                    oracle_action_type = oracle.action_type if oracle.action_type == "MERGE" else "NOT_MERGE"
                    oracle_reasoning = oracle.gt_reasoning

                context_summary = {
                    "cluster_id": small_cid,
                    "merge_target_id": large_cid,
                    "n_small": int(small_info["n_spikes"]),
                    "n_large": int(large_info["n_spikes"]),
                }

                teacher_feedback = self.teacher.get_feedback(
                    channel_id=channel_id,
                    step=current_step,
                    action_type="merge",
                    context_summary=context_summary,
                    student_action=student_resp.get("action", ""),
                    student_rationale=student_resp.get("rationale", ""),
                    gt_action=oracle_action_type,
                    gt_reasoning=oracle_reasoning,
                    images=[],
                    merge_target_id=large_cid,
                    output_dir=output_dir / "teacher_outputs",
                )

                step = TrajectoryStep(
                    step=current_step,
                    channel_id=channel_id,
                    setting_id=setting_id,
                    cluster_id=small_cid,
                    action_phase="phase2",
                    target_id=large_cid,
                    gt_action=oracle_action_type,
                    gt_reasoning=oracle_reasoning,
                    student_action=student_resp.get("action", ""),
                    student_rationale=student_resp.get("rationale", ""),
                    student_model=self.student.student_model,
                    teacher_feedback=teacher_feedback,
                    teacher_model=self.teacher.teacher_model,
                    n_spikes=int(small_info["n_spikes"]),
                    n_active_clusters=len(manager.get_active_clusters()),
                )
                steps.append(step)

                if self.rag_memory is not None:
                    rag_entry = build_phase2_memory_entry(
                        channel_id=channel_id,
                        step=current_step,
                        cluster_id=small_cid,
                        target_id=large_cid,
                        small_waveforms=small_info["waveforms"],
                        small_spike_times=small_info["spike_times"],
                        large_waveforms=large_info["waveforms"],
                        large_spike_times=large_info["spike_times"],
                        gt_action=oracle_action_type,
                        gt_reasoning=oracle_reasoning,
                        prompt_text=str(student_resp.get("prompt_text", "")),
                        image_paths=[str(p) for p in student_resp.get("image_paths", [])],
                    )
                    self.rag_memory.add(rag_entry)

                step_idx += 1

                if oracle_action_type == "MERGE":
                    manager.merge_clusters([small_cid, large_cid], target_id=large_cid)
                    break   # small_cid is now merged; move to next small cluster

        # --- Save ---
        output_dir.mkdir(parents=True, exist_ok=True)
        save_trajectory(steps, out_file)
        summary = trajectory_summary(steps)
        with open(output_dir / "trajectory_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"  [done] {setting_id}/{channel_id}: {len(steps)} steps, "
            f"student accuracy={summary['student_action_accuracy']:.2f}"
        )
        return steps


def _apply_oracle(manager: ClusterManager, oracle) -> None:
    """Apply the oracle action to advance the manager state."""
    if oracle.action_type == "SPLIT":
        try:
            manager.split_last_merge(oracle.cluster_id)
        except Exception:
            manager.split_by_overclusters(oracle.cluster_id)
    elif oracle.action_type == "DISCARD":
        manager.discard_cluster(oracle.cluster_id)
    elif oracle.action_type == "MERGE" and oracle.target_id is not None:
        manager.merge_clusters([oracle.cluster_id, oracle.target_id], target_id=oracle.target_id)
