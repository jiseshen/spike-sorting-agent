"""
Build supervised finetune data from MATLAB spikes + human action sheets.

Pipeline (aligned to agent workflow):
1. Load MATLAB channel data and action sheet
2. Prefilter tiny overclusters
3. Phase 0 auto-discard tiny hierarchy clusters
4. Replay expert split/discard actions (simulate iterative split process)
5. Add synthetic KEEP samples for terminal split-stage clusters
6. Construct merge-stage positives from expert merge actions
7. Add merge-stage negative samples (NOT_MERGE), sampled as ~1/8 of expert steps

Outputs:
- output/finetune_dataset/finetune_dataset_mixed.jsonl
- output/finetune_dataset/finetune_dataset_summary.json
- output/finetune_dataset/images/<sample_id>/*.png
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.io.matlab_loader import load_matlab_spikes
from src.cluster.manager import ClusterManager
from src.cluster.auto_filter import automatic_size_filter
from src.agent.runner import (
    compute_merged_isi_violation_rate,
    compute_waveform_correlation,
    create_aggregation_tree_image,
    create_amplitude_distribution_image,
    create_isi_histogram_image,
    create_waveform_comparison_image,
    create_waveform_overlay_image,
)


@dataclass
class ActionStep:
    """Parsed action sheet step."""

    raw_action: str
    reasoning: str
    action_type: str  # split | discard | merge
    cluster_id: int
    target_id: Optional[int] = None


def parse_action(action_str: str) -> ActionStep:
    """Parse one action string from action sheet."""
    raw = action_str.strip().strip("'\"")

    m = re.match(r"s\s+(\d+)$", raw)
    if m:
        return ActionStep(raw_action=raw, reasoning="", action_type="split", cluster_id=int(m.group(1)))

    m = re.match(r"m\s+(\d+)\s+(\d+)$", raw)
    if m:
        target = int(m.group(1))
        source = int(m.group(2))
        if target == 0:
            return ActionStep(raw_action=raw, reasoning="", action_type="discard", cluster_id=source)
        return ActionStep(
            raw_action=raw, reasoning="", action_type="merge", cluster_id=source, target_id=target
        )

    raise ValueError(f"Unrecognized action format: {action_str}")


def load_action_sheet(channel: str) -> List[ActionStep]:
    """Load and parse channel action sheet."""
    path = Path(f"data/action_sheets/{channel}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Action sheet not found: {path}")

    steps: List[ActionStep] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row["Actions"].strip()
            reasoning = row.get("Action Reasoning", "").strip().strip("'\"")
            step = parse_action(action)
            step.reasoning = reasoning
            steps.append(step)
    return steps


def compute_isi_violation_rate(spike_times: np.ndarray, refractory_s: float = 0.002) -> float:
    """Compute ISI violation rate for one cluster."""
    if spike_times.size < 2:
        return 0.0
    isis = np.diff(np.sort(spike_times))
    return float(np.mean(isis < refractory_s))


def compute_amplitude_cv(waveforms: np.ndarray) -> float:
    """Compute amplitude CV from peak-to-trough amplitudes."""
    if waveforms.shape[0] == 0:
        return 0.0
    amps = np.max(waveforms, axis=1) - np.min(waveforms, axis=1)
    mean_amp = float(np.mean(amps))
    if mean_amp <= 1e-12:
        return 0.0
    return float(np.std(amps) / mean_amp)


def prefilter_small_overclusters(
    overcluster_assigns: np.ndarray, hierarchy_assigns: np.ndarray, min_spikes: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Set tiny overclusters to noise (0) in both overcluster and hierarchy labels."""
    over = overcluster_assigns.copy()
    hier = hierarchy_assigns.copy()

    valid = over[over > 0]
    unique_ids, counts = np.unique(valid, return_counts=True)
    tiny_ids = unique_ids[counts < min_spikes]

    if tiny_ids.size == 0:
        return over, hier, {"n_tiny_overclusters": 0, "n_spikes_filtered": 0, "tiny_overcluster_ids": []}

    tiny_mask = np.isin(over, tiny_ids)
    n_spikes_filtered = int(np.sum(tiny_mask))

    over[tiny_mask] = 0
    hier[tiny_mask] = 0

    meta = {
        "n_tiny_overclusters": int(tiny_ids.size),
        "n_spikes_filtered": n_spikes_filtered,
        "tiny_overcluster_ids": [int(x) for x in tiny_ids.tolist()],
    }
    return over, hier, meta


class FineTuneDatasetBuilder:
    """Builds finetune samples with prompts, images, metrics, and targets."""

    def __init__(
        self,
        output_dir: Path,
        seed: int,
        small_cluster_threshold: int,
        final_minimum_threshold: int,
        profile: str = "legacy_mixed",
        expert_only: bool = False,
        fixed_target_format: str = "auto",
        output_jsonl_name: str = "finetune_dataset_mixed.jsonl",
        summary_name: str = "finetune_dataset_summary.json",
    ):
        self.output_dir = output_dir
        self.image_root = output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(seed)
        self.small_cluster_threshold = small_cluster_threshold
        self.final_minimum_threshold = final_minimum_threshold
        self.profile = profile
        self.expert_only = expert_only
        self.fixed_target_format = fixed_target_format
        self.output_jsonl_name = output_jsonl_name
        self.summary_name = summary_name

        self.samples: List[Dict[str, Any]] = []
        self._sample_index = 0

    def _next_sample_id(self, channel: str) -> str:
        self._sample_index += 1
        return f"{channel}_{self._sample_index:06d}"

    def _save_b64_image(self, sample_id: str, name: str, img_b64: str) -> str:
        sample_dir = self.image_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / f"{name}.png"
        path.write_bytes(base64.b64decode(img_b64))
        return str(path.relative_to(self.output_dir))

    def _pick_target_format(self) -> str:
        if self.fixed_target_format != "auto":
            return self.fixed_target_format
        if self.profile == "gemma4_train_reasoned":
            return "reasoned_json"
        if self.profile == "gemma4_action_eval":
            return "action_label"
        return "react_json" if self.rng.random() < 0.5 else "natural_language"

    def _format_target(self, fmt: str, reason: str, action: str) -> str:
        if fmt in ("react_json", "reasoned_json"):
            return json.dumps({"reason": reason, "action": action}, ensure_ascii=False)
        if fmt == "action_label":
            return action
        if fmt == "reason_action":
            return f"Reason: {reason}\nAction: {action}"
        return f"<reason>{reason}</reason>\nAction:{action}"

    @staticmethod
    def _allowed_actions(stage: str) -> List[str]:
        if stage == "merge":
            return ["MERGE", "NOT_MERGE", "DISCARD"]
        return ["KEEP", "DISCARD", "SPLIT"]

    def _should_keep_source(self, source: str) -> bool:
        if not self.expert_only:
            return True
        return source.startswith("expert_")

    def _append_sample(self, sample: Optional[Dict[str, Any]]) -> None:
        if sample is None:
            return
        source = str(sample.get("source", ""))
        if not self._should_keep_source(source):
            return
        self.samples.append(sample)

    def _keep_reason(self, cluster_id: int, n_spikes: int) -> str:
        if n_spikes >= self.final_minimum_threshold:
            return (
                f"Cluster {cluster_id} shows a valid neuronal waveform shape, reasonably consistent waveform family, "
                "and acceptable ISI behavior. Keep it as a standalone neuronal unit."
            )
        return (
            f"Cluster {cluster_id} shows a valid neuronal waveform shape, reasonably consistent waveform family, "
            "and acceptable ISI behavior. Keep it for now and consider merge with similar clusters to reach "
            f"the final size target (>= {self.final_minimum_threshold} spikes)."
        )

    @staticmethod
    def _not_merge_reason(small_id: int, large_id: int, corr: float, merged_isi: float) -> str:
        return (
            f"Do not merge cluster {small_id} into {large_id}: waveform similarity is insufficient "
            f"(corr={corr:.3f}) or merge quality is questionable (merged ISI={merged_isi:.2%}). "
            "Keeping them separate avoids mixing distinct units."
        )

    def _build_split_prompt(self, cluster_id: int, metrics: Dict[str, Any], fmt: str) -> str:
        if self.profile == "gemma4_action_eval":
            return (
                "You are in SPLIT stage of spike sorting curation.\n"
                "Action space: KEEP, DISCARD, SPLIT.\n"
                "Use all attached images and metrics. Decide only the final action.\n"
                f"\nCluster {cluster_id} metrics:\n"
                f"- n_spikes: {metrics['n_spikes']}\n"
                f"- n_overclusters: {metrics['n_overclusters']}\n"
                f"- isi_violation_rate: {metrics['isi_violation_rate']:.2%}\n"
                f"- amplitude_cv: {metrics['amplitude_cv']:.3f}\n"
                "\nOutput exactly one token: KEEP or DISCARD or SPLIT.\n"
                "Do not output rationale, explanation, JSON, markdown, or extra text."
            )

        if self.profile == "gemma4_train_reasoned":
            return (
                "You are in SPLIT stage of spike sorting curation.\n"
                "Action space: KEEP, DISCARD, SPLIT.\n"
                "Use all attached images and metrics to determine the best action.\n"
                f"\nCluster {cluster_id} metrics:\n"
                f"- n_spikes: {metrics['n_spikes']}\n"
                f"- n_overclusters: {metrics['n_overclusters']}\n"
                f"- isi_violation_rate: {metrics['isi_violation_rate']:.2%}\n"
                f"- amplitude_cv: {metrics['amplitude_cv']:.3f}\n"
                "\nReturn only final answer JSON (no hidden reasoning): "
                '{"reason":"brief rationale","action":"KEEP|DISCARD|SPLIT"}'
            )

        if fmt == "react_json":
            out_spec = (
                'Output format (STRICT): one-line valid JSON only: {"reason":"...","action":"KEEP|DISCARD|SPLIT"}. '
                "No markdown, no code fence, no extra keys, no extra text."
            )
        elif fmt == "reason_action":
            out_spec = (
                "Output format (STRICT): exactly two lines:\n"
                "Reason: ...\n"
                "Action: KEEP|DISCARD|SPLIT\n"
                "No extra lines, no extra text."
            )
        else:
            out_spec = (
                "Output format (STRICT): exactly two lines:\n"
                "<reason>...</reason>\n"
                "Action:KEEP|DISCARD|SPLIT\n"
                "No extra lines, no extra text."
            )
        return (
            "You are in SPLIT stage of spike sorting curation.\n"
            "Action space: KEEP, DISCARD, SPLIT.\n"
            "Decision rule:\n"
            "- DISCARD if waveform is clearly non-neuronal/noisy.\n"
            "- SPLIT if multiple waveform families or mixed-unit pattern is visible.\n"
            "- KEEP if waveform is neuronal and internally consistent.\n"
            f"\nCluster {cluster_id} metrics:\n"
            f"- n_spikes: {metrics['n_spikes']}\n"
            f"- n_overclusters: {metrics['n_overclusters']}\n"
            f"- isi_violation_rate: {metrics['isi_violation_rate']:.2%}\n"
            f"- amplitude_cv: {metrics['amplitude_cv']:.3f}\n"
            "\nUse all attached images (waveform overlay, ISI histogram, amplitude distribution, aggregation tree).\n"
            "Action must be exactly one of: KEEP, DISCARD, SPLIT.\n"
            f"{out_spec}"
        )

    def _build_merge_prompt(self, small_id: int, large_id: int, metrics: Dict[str, Any], fmt: str) -> str:
        if self.profile == "gemma4_action_eval":
            return (
                "You are in MERGE stage of spike sorting curation.\n"
                "Action space: MERGE, NOT_MERGE, DISCARD.\n"
                "Use all attached images and metrics. Decide only the final action.\n"
                f"\nCandidate merge: small {small_id} -> large {large_id}\n"
                f"- small_n_spikes: {metrics['small_n_spikes']}\n"
                f"- large_n_spikes: {metrics['large_n_spikes']}\n"
                f"- small_isi_rate: {metrics['small_isi_rate']:.2%}\n"
                f"- large_isi_rate: {metrics['large_isi_rate']:.2%}\n"
                f"- waveform_correlation: {metrics['waveform_correlation']:.3f}\n"
                f"- merged_isi_rate: {metrics['merged_isi_rate']:.2%}\n"
                "\nOutput exactly one token: MERGE or NOT_MERGE or DISCARD.\n"
                "Do not output rationale, explanation, JSON, markdown, or extra text."
            )

        if self.profile == "gemma4_train_reasoned":
            return (
                "You are in MERGE stage of spike sorting curation.\n"
                "Action space: MERGE, NOT_MERGE, DISCARD.\n"
                "Use all attached images and metrics to determine the best action.\n"
                f"\nCandidate merge: small {small_id} -> large {large_id}\n"
                f"- small_n_spikes: {metrics['small_n_spikes']}\n"
                f"- large_n_spikes: {metrics['large_n_spikes']}\n"
                f"- small_isi_rate: {metrics['small_isi_rate']:.2%}\n"
                f"- large_isi_rate: {metrics['large_isi_rate']:.2%}\n"
                f"- waveform_correlation: {metrics['waveform_correlation']:.3f}\n"
                f"- merged_isi_rate: {metrics['merged_isi_rate']:.2%}\n"
                "\nReturn only final answer JSON (no hidden reasoning): "
                '{"reason":"brief rationale","action":"MERGE|NOT_MERGE|DISCARD"}'
            )

        if fmt == "react_json":
            out_spec = (
                'Output format (STRICT): one-line valid JSON only: {"reason":"...","action":"MERGE|NOT_MERGE|DISCARD"}. '
                "No markdown, no code fence, no extra keys, no extra text."
            )
        elif fmt == "reason_action":
            out_spec = (
                "Output format (STRICT): exactly two lines:\n"
                "Reason: ...\n"
                "Action: MERGE|NOT_MERGE|DISCARD\n"
                "No extra lines, no extra text."
            )
        else:
            out_spec = (
                "Output format (STRICT): exactly two lines:\n"
                "<reason>...</reason>\n"
                "Action:MERGE|NOT_MERGE|DISCARD\n"
                "No extra lines, no extra text."
            )
        return (
            "You are in MERGE stage of spike sorting curation.\n"
            "Action space: MERGE, NOT_MERGE, DISCARD.\n"
            "Decision rule:\n"
            "- MERGE if the small cluster is neuronal and compatible with the large cluster.\n"
            "- NOT_MERGE if clusters are both plausible but not compatible.\n"
            "- DISCARD if small cluster itself is invalid/noisy.\n"
            f"\nCandidate merge: small {small_id} -> large {large_id}\n"
            f"- small_n_spikes: {metrics['small_n_spikes']}\n"
            f"- large_n_spikes: {metrics['large_n_spikes']}\n"
            f"- small_isi_rate: {metrics['small_isi_rate']:.2%}\n"
            f"- large_isi_rate: {metrics['large_isi_rate']:.2%}\n"
            f"- waveform_correlation: {metrics['waveform_correlation']:.3f}\n"
            f"- merged_isi_rate: {metrics['merged_isi_rate']:.2%}\n"
            "\nUse all attached images (small waveform, large waveform, merged ISI histogram).\n"
            "Action must be exactly one of: MERGE, NOT_MERGE, DISCARD.\n"
            f"{out_spec}"
        )

    def _build_split_sample(
        self,
        channel: str,
        manager: ClusterManager,
        fs: float,
        cluster_id: int,
        label_action: str,
        label_reason: str,
        source: str,
        expert_action_raw: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        info = manager.get_cluster_info(cluster_id)
        if info is None or info["n_spikes"] == 0:
            return None

        waveforms = info["waveforms"]
        spike_times = info["spike_times"]
        overclusters = info["overclusters"]

        metrics = {
            "n_spikes": int(info["n_spikes"]),
            "n_overclusters": int(info["n_overclusters"]),
            "isi_violation_rate": compute_isi_violation_rate(spike_times),
            "amplitude_cv": compute_amplitude_cv(waveforms),
        }

        sample_id = self._next_sample_id(channel)
        fmt = self._pick_target_format()

        wf_img = create_waveform_overlay_image(waveforms=waveforms, cluster_id=cluster_id, sampling_rate=fs)
        isi_img = create_isi_histogram_image(spike_times=spike_times, cluster_id=cluster_id)
        amp_img = create_amplitude_distribution_image(waveforms=waveforms, cluster_id=cluster_id)
        tree_img = create_aggregation_tree_image(
            hierarchy_tree=manager.hierarchy_tree,
            overcluster_composition=overclusters,
            cluster_id=cluster_id,
        )

        image_paths = {
            "waveform_overlay": self._save_b64_image(sample_id, "waveform_overlay", wf_img),
            "isi_histogram": self._save_b64_image(sample_id, "isi_histogram", isi_img),
            "amplitude_distribution": self._save_b64_image(sample_id, "amplitude_distribution", amp_img),
            "aggregation_tree": self._save_b64_image(sample_id, "aggregation_tree", tree_img),
        }

        prompt = self._build_split_prompt(cluster_id=cluster_id, metrics=metrics, fmt=fmt)
        target = self._format_target(fmt=fmt, reason=label_reason, action=label_action)

        return {
            "id": sample_id,
            "channel": channel,
            "stage": "split",
            "profile": self.profile,
            "output_mode": "action_only" if self.profile == "gemma4_action_eval" else "reasoned",
            "allowed_actions": self._allowed_actions("split"),
            "source": source,
            "expert_action_raw": expert_action_raw,
            "prompt": prompt,
            "images": image_paths,
            "metrics": metrics,
            "target_format": fmt,
            "target": target,
            "label_action": label_action,
            "label_reason": label_reason,
        }

    def _build_merge_sample(
        self,
        channel: str,
        manager: ClusterManager,
        fs: float,
        small_cluster_id: int,
        large_cluster_id: int,
        label_action: str,
        label_reason: str,
        source: str,
        expert_action_raw: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        small_info = manager.get_cluster_info(small_cluster_id)
        large_info = manager.get_cluster_info(large_cluster_id)
        if small_info is None or large_info is None:
            return None

        small_wf = small_info["waveforms"]
        large_wf = large_info["waveforms"]
        small_ts = small_info["spike_times"]
        large_ts = large_info["spike_times"]

        corr = compute_waveform_correlation(small_wf, large_wf)
        merged_isi = compute_merged_isi_violation_rate(small_ts, large_ts)
        small_isi = compute_isi_violation_rate(small_ts)
        large_isi = compute_isi_violation_rate(large_ts)

        metrics = {
            "small_n_spikes": int(small_info["n_spikes"]),
            "large_n_spikes": int(large_info["n_spikes"]),
            "small_isi_rate": small_isi,
            "large_isi_rate": large_isi,
            "waveform_correlation": float(corr),
            "merged_isi_rate": float(merged_isi),
        }

        sample_id = self._next_sample_id(channel)
        fmt = self._pick_target_format()

        small_img, large_img, merged_isi_img = create_waveform_comparison_image(
            small_waveforms=small_wf,
            large_waveforms=large_wf,
            small_spike_times=small_ts,
            large_spike_times=large_ts,
            small_cluster_id=small_cluster_id,
            large_cluster_id=large_cluster_id,
            sampling_rate=fs,
        )

        image_paths = {
            "small_waveform_overlay": self._save_b64_image(sample_id, "small_waveform_overlay", small_img),
            "large_waveform_overlay": self._save_b64_image(sample_id, "large_waveform_overlay", large_img),
            "merged_isi_histogram": self._save_b64_image(sample_id, "merged_isi_histogram", merged_isi_img),
        }

        prompt = self._build_merge_prompt(
            small_id=small_cluster_id,
            large_id=large_cluster_id,
            metrics=metrics,
            fmt=fmt,
        )
        target = self._format_target(fmt=fmt, reason=label_reason, action=label_action)

        return {
            "id": sample_id,
            "channel": channel,
            "stage": "merge",
            "profile": self.profile,
            "output_mode": "action_only" if self.profile == "gemma4_action_eval" else "reasoned",
            "allowed_actions": self._allowed_actions("merge"),
            "source": source,
            "expert_action_raw": expert_action_raw,
            "prompt": prompt,
            "images": image_paths,
            "metrics": metrics,
            "target_format": fmt,
            "target": target,
            "label_action": label_action,
            "label_reason": label_reason,
            "small_cluster_id": int(small_cluster_id),
            "large_cluster_id": int(large_cluster_id),
        }

    def build_for_channel(
        self,
        channel: str,
        min_overcluster_spikes: int,
        auto_discard_threshold: int,
        max_expert_actions: int,
    ) -> Dict[str, Any]:
        """Build channel dataset and append to global sample list."""
        data = load_matlab_spikes(f"data/{channel}_spikes.mat")
        fs = float(data["Fs"])

        over_f, hier_f, prefilter_meta = prefilter_small_overclusters(
            overcluster_assigns=np.asarray(data["overcluster_assigns"]),
            hierarchy_assigns=np.asarray(data["hierarchy_assigns"]),
            min_spikes=min_overcluster_spikes,
        )

        manager = ClusterManager(
            initial_assigns=hier_f,
            overcluster_assigns=over_f,
            hierarchy_tree=np.asarray(data["hierarchy_tree"]).copy(),
            spike_times=np.asarray(data["spiketimes"]),
            waveforms=np.asarray(data["waveforms"]),
        )

        kept, auto_discard_actions = automatic_size_filter(manager.assigns, threshold=auto_discard_threshold)
        for a in auto_discard_actions:
            try:
                manager.discard_cluster(a.cluster_id)
            except Exception:
                pass

        steps = load_action_sheet(channel)
        if max_expert_actions > 0:
            steps = steps[:max_expert_actions]

        split_steps = [s for s in steps if s.action_type in ("split", "discard")]
        merge_steps = [s for s in steps if s.action_type == "merge"]

        keep_emitted: set[int] = set()
        channel_samples_start = len(self.samples)

        # --------------------
        # Split stage replay
        # --------------------
        for i, step in enumerate(split_steps):
            cid = step.cluster_id
            info = manager.get_cluster_info(cid)
            if info is None or info["n_spikes"] == 0:
                continue

            if step.action_type == "split":
                sample = self._build_split_sample(
                    channel=channel,
                    manager=manager,
                    fs=fs,
                    cluster_id=cid,
                    label_action="SPLIT",
                    label_reason=step.reasoning or "Expert split decision based on waveform mixture and hierarchy.",
                    source="expert_split",
                    expert_action_raw=step.raw_action,
                )
                self._append_sample(sample)

                try:
                    new_ids = manager.split_last_merge(cid)
                except Exception:
                    continue

                # Add synthetic KEEP for terminal children (no future split/discard action on that cluster).
                for child_id in sorted(set(int(x) for x in new_ids)):
                    child_info = manager.get_cluster_info(child_id)
                    if child_info is None or child_info["n_spikes"] == 0:
                        continue
                    if child_id in keep_emitted:
                        continue

                    has_future_action = any(s.cluster_id == child_id for s in split_steps[i + 1 :])
                    if has_future_action:
                        continue

                    keep_sample = self._build_split_sample(
                        channel=channel,
                        manager=manager,
                        fs=fs,
                        cluster_id=child_id,
                        label_action="KEEP",
                        label_reason=self._keep_reason(child_id, int(child_info["n_spikes"])),
                        source="synthetic_keep_terminal",
                    )
                    if keep_sample is not None:
                        self._append_sample(keep_sample)
                        keep_emitted.add(child_id)

            elif step.action_type == "discard":
                sample = self._build_split_sample(
                    channel=channel,
                    manager=manager,
                    fs=fs,
                    cluster_id=cid,
                    label_action="DISCARD",
                    label_reason=step.reasoning or "Expert discarded this cluster as non-neuronal/noisy.",
                    source="expert_discard",
                    expert_action_raw=step.raw_action,
                )
                self._append_sample(sample)

                try:
                    manager.discard_cluster(cid)
                except Exception:
                    continue

        # Add KEEP for any remaining active clusters without explicit terminal label.
        for cid in sorted(manager.get_active_clusters()):
            if cid in keep_emitted:
                continue
            keep_sample = self._build_split_sample(
                channel=channel,
                manager=manager,
                fs=fs,
                cluster_id=cid,
                label_action="KEEP",
                label_reason=self._keep_reason(cid, int(np.sum(manager.assigns == cid))),
                source="synthetic_keep_remaining",
            )
            if keep_sample is not None:
                self._append_sample(keep_sample)
                keep_emitted.add(cid)

        # --------------------
        # Merge negatives (NOT_MERGE)
        # --------------------
        n_negatives = max(1, len(steps) // 8)
        expert_merge_pairs = {(s.cluster_id, s.target_id) for s in merge_steps if s.target_id is not None}

        active = sorted(manager.get_active_clusters())
        sizes = {cid: int(np.sum(manager.assigns == cid)) for cid in active}
        small_ids = [cid for cid in active if sizes[cid] < self.small_cluster_threshold]
        large_ids = [cid for cid in active if sizes[cid] >= self.small_cluster_threshold]

        candidate_pairs: List[Tuple[int, int]] = []
        for s in small_ids:
            for l in large_ids:
                if s == l:
                    continue
                if (s, l) in expert_merge_pairs:
                    continue
                candidate_pairs.append((s, l))

        # Fallback candidates if threshold split yields no pairs.
        if not candidate_pairs and len(active) >= 2:
            for s in active:
                for l in active:
                    if s == l:
                        continue
                    if (s, l) in expert_merge_pairs:
                        continue
                    candidate_pairs.append((s, l))

        if candidate_pairs:
            n_take = min(n_negatives, len(candidate_pairs))
            chosen_idx = self.rng.choice(len(candidate_pairs), size=n_take, replace=False)
            for idx in chosen_idx:
                s, l = candidate_pairs[int(idx)]
                s_info = manager.get_cluster_info(s)
                l_info = manager.get_cluster_info(l)
                if s_info is None or l_info is None:
                    continue
                corr = compute_waveform_correlation(s_info["waveforms"], l_info["waveforms"])
                merged_isi = compute_merged_isi_violation_rate(s_info["spike_times"], l_info["spike_times"])
                reason = self._not_merge_reason(s, l, corr=corr, merged_isi=merged_isi)

                neg_sample = self._build_merge_sample(
                    channel=channel,
                    manager=manager,
                    fs=fs,
                    small_cluster_id=s,
                    large_cluster_id=l,
                    label_action="NOT_MERGE",
                    label_reason=reason,
                    source="synthetic_merge_negative",
                )
                self._append_sample(neg_sample)

        # --------------------
        # Expert merge positives
        # --------------------
        for step in merge_steps:
            source = step.cluster_id
            target = step.target_id
            if target is None:
                continue

            s_info = manager.get_cluster_info(source)
            t_info = manager.get_cluster_info(target)
            if s_info is None or t_info is None:
                continue

            merge_sample = self._build_merge_sample(
                channel=channel,
                manager=manager,
                fs=fs,
                small_cluster_id=source,
                large_cluster_id=target,
                label_action="MERGE",
                label_reason=step.reasoning or "Expert merge decision based on similarity and post-merge quality.",
                source="expert_merge",
                expert_action_raw=step.raw_action,
            )
            self._append_sample(merge_sample)

            try:
                manager.merge_clusters([source, target], target_id=target)
            except Exception:
                continue

        channel_samples = self.samples[channel_samples_start:]
        counts_by_action: Dict[str, int] = {}
        for x in channel_samples:
            counts_by_action[x["label_action"]] = counts_by_action.get(x["label_action"], 0) + 1

        return {
            "channel": channel,
            "n_samples": len(channel_samples),
            "counts_by_action": counts_by_action,
            "prefilter": prefilter_meta,
            "n_phase0_auto_discard": len(auto_discard_actions),
            "n_input_actions": len(steps),
            "n_split_actions": len(split_steps),
            "n_merge_actions": len(merge_steps),
            "n_kept_after_split_stage": len(keep_emitted),
        }

    def save(self) -> Dict[str, Any]:
        """Save dataset jsonl and summary."""
        dataset_path = self.output_dir / self.output_jsonl_name
        with open(dataset_path, "w") as f:
            for row in self.samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        by_action: Dict[str, int] = {}
        by_stage: Dict[str, int] = {}
        by_format: Dict[str, int] = {}
        by_output_mode: Dict[str, int] = {}
        for row in self.samples:
            by_action[row["label_action"]] = by_action.get(row["label_action"], 0) + 1
            by_stage[row["stage"]] = by_stage.get(row["stage"], 0) + 1
            by_format[row["target_format"]] = by_format.get(row["target_format"], 0) + 1
            mode = str(row.get("output_mode", "reasoned"))
            by_output_mode[mode] = by_output_mode.get(mode, 0) + 1

        summary = {
            "profile": self.profile,
            "expert_only": self.expert_only,
            "fixed_target_format": self.fixed_target_format,
            "n_samples_total": len(self.samples),
            "counts_by_action": by_action,
            "counts_by_stage": by_stage,
            "counts_by_target_format": by_format,
            "counts_by_output_mode": by_output_mode,
            "dataset_path": str(dataset_path),
        }
        with open(self.output_dir / self.summary_name, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build finetune dataset from action sheets.")
    parser.add_argument("--channels", nargs="+", default=["CH3", "CH20", "CH30", "CH31"])
    parser.add_argument("--output-dir", default="output/finetune_dataset")
    parser.add_argument(
        "--profile",
        default="legacy_mixed",
        choices=["legacy_mixed", "gemma4_train_reasoned", "gemma4_action_eval"],
        help="legacy_mixed keeps existing random dual-format behavior.",
    )
    parser.add_argument(
        "--expert-only",
        action="store_true",
        help="Keep only expert_* samples in final JSONL (drops synthetic samples).",
    )
    parser.add_argument(
        "--output-jsonl-name",
        default="",
        help="Optional output JSONL file name. Defaults by profile.",
    )
    parser.add_argument(
        "--summary-name",
        default="",
        help="Optional summary JSON file name. Defaults by profile.",
    )
    parser.add_argument(
        "--fixed-target-format",
        default="auto",
        choices=["auto", "react_json", "natural_language", "reason_action", "reasoned_json", "action_label"],
        help="Force a single output target format for all samples. auto=profile default behavior.",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Defaults aligned with src/pipeline/pure.py thresholds.
    parser.add_argument("--min-overcluster-spikes", type=int, default=500)
    parser.add_argument("--auto-discard-threshold", type=int, default=500)
    parser.add_argument("--small-cluster-threshold", type=int, default=4000)
    parser.add_argument("--final-minimum-threshold", type=int, default=5000)
    parser.add_argument(
        "--max-expert-actions",
        type=int,
        default=0,
        help="For debugging. 0 means no limit.",
    )
    args = parser.parse_args()

    default_jsonl_by_profile = {
        "legacy_mixed": "finetune_dataset_mixed.jsonl",
        "gemma4_train_reasoned": "train_reasoned.jsonl",
        "gemma4_action_eval": "eval_action_only.jsonl",
    }
    default_summary_by_profile = {
        "legacy_mixed": "finetune_dataset_summary.json",
        "gemma4_train_reasoned": "train_reasoned_summary.json",
        "gemma4_action_eval": "eval_action_only_summary.json",
    }

    builder = FineTuneDatasetBuilder(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        small_cluster_threshold=args.small_cluster_threshold,
        final_minimum_threshold=args.final_minimum_threshold,
        profile=args.profile,
        expert_only=args.expert_only,
        fixed_target_format=args.fixed_target_format,
        output_jsonl_name=(args.output_jsonl_name or default_jsonl_by_profile[args.profile]),
        summary_name=(args.summary_name or default_summary_by_profile[args.profile]),
    )

    channel_reports: List[Dict[str, Any]] = []
    for ch in args.channels:
        report = builder.build_for_channel(
            channel=ch,
            min_overcluster_spikes=args.min_overcluster_spikes,
            auto_discard_threshold=args.auto_discard_threshold,
            max_expert_actions=args.max_expert_actions,
        )
        channel_reports.append(report)
        print(f"[{ch}] samples={report['n_samples']} actions={report['counts_by_action']}")

    summary = builder.save()
    summary["channel_reports"] = channel_reports
    with open(Path(args.output_dir) / builder.summary_name, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"- Profile: {args.profile}")
    print(f"- Expert only: {args.expert_only}")
    print(f"- Total samples: {summary['n_samples_total']}")
    print(f"- By action: {summary['counts_by_action']}")
    print(f"- By format: {summary['counts_by_target_format']}")
    print(f"- By output_mode: {summary['counts_by_output_mode']}")
    print(f"- Dataset: {summary['dataset_path']}")


if __name__ == "__main__":
    main()
