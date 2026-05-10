"""Unit Test: VLM Decision Accuracy vs Human Ground Truth.

Supports A/B comparison between:
- no_rag: original decision path
- rag: continual RAG baseline with structured memory retrieval

Workflow:
1. Load channel data + human action sheet (data/action_sheets/CHX.csv)
2. Phase 0 auto-discard small clusters (not counted in GT comparison)
3. Replay human actions step-by-step
4. For RAG mode: retrieve similar past cases before each VLM call, then append
   current evidence + GT action/reasoning to memory
5. Report per-mode and comparative metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.rag_memory import (
    ContinualRAGMemory,
    build_phase1_memory_entry,
    build_phase2_memory_entry,
)
from src.agent.runner import (
    vlm_phase1_cluster_decision,
    vlm_phase2_merge_decision,
)
from src.cluster.auto_filter import automatic_size_filter
from src.cluster.features import ClusterFeatures
from src.cluster.manager import ClusterManager
from src.io.matlab_loader import load_matlab_spikes


DEFAULT_CHANNELS = ["CH3", "CH20", "CH30", "CH31"]


@dataclass
class ComparisonResult:
    """Record of one VLM vs human comparison."""

    channel: str
    step: int
    action_type: str
    cluster_id: int
    merge_target_id: Optional[int]
    human_decision: str
    vlm_decision: str
    match: bool
    human_reasoning: str
    vlm_reasoning: str


@dataclass
class ChannelTestResults:
    """Aggregated comparisons for one channel."""

    channel: str
    comparisons: List[ComparisonResult] = field(default_factory=list)
    temp_smoke_truncated: bool = False

    def add_comparison(self, comp: ComparisonResult) -> None:
        self.comparisons.append(comp)

    def get_metrics(self) -> Dict[str, Any]:
        if not self.comparisons:
            return {
                "split_recall": 0.0,
                "discard_recall": 0.0,
                "merge_recall": 0.0,
                "overall_accuracy": 0.0,
                "n_split": 0,
                "n_discard": 0,
                "n_merge": 0,
                "n_total": 0,
                "split_correct": 0,
                "discard_correct": 0,
                "merge_correct": 0,
                "total_correct": 0,
            }

        split_comps = [c for c in self.comparisons if c.action_type == "split"]
        discard_comps = [c for c in self.comparisons if c.action_type == "discard"]
        merge_comps = [c for c in self.comparisons if c.action_type == "merge"]

        split_correct = sum(1 for c in split_comps if c.match)
        discard_correct = sum(1 for c in discard_comps if c.match)
        merge_correct = sum(1 for c in merge_comps if c.match)
        total_correct = sum(1 for c in self.comparisons if c.match)

        return {
            "split_recall": split_correct / len(split_comps) if split_comps else 0.0,
            "discard_recall": discard_correct / len(discard_comps) if discard_comps else 0.0,
            "merge_recall": merge_correct / len(merge_comps) if merge_comps else 0.0,
            "overall_accuracy": total_correct / len(self.comparisons),
            "n_split": len(split_comps),
            "n_discard": len(discard_comps),
            "n_merge": len(merge_comps),
            "n_total": len(self.comparisons),
            "split_correct": split_correct,
            "discard_correct": discard_correct,
            "merge_correct": merge_correct,
            "total_correct": total_correct,
        }


def parse_action(action_str: str) -> Tuple[str, int, Optional[int]]:
    action_str = action_str.strip().strip("'\"")

    split_match = re.match(r"s\s+(\d+)", action_str)
    if split_match:
        return ("split", int(split_match.group(1)), None)

    merge_match = re.match(r"m\s+(\d+)\s+(\d+)", action_str)
    if merge_match:
        target = int(merge_match.group(1))
        source = int(merge_match.group(2))
        if target == 0:
            return ("discard", source, None)
        return ("merge", source, target)

    raise ValueError(f"Cannot parse action: {action_str}")


def load_action_sheet(channel: str) -> List[Tuple[str, str]]:
    csv_path = Path(f"data/action_sheets/{channel}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Action sheet not found: {csv_path}")

    actions: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row.get("Actions", "").strip()
            reasoning = row.get("Action Reasoning", "").strip()
            if action:
                actions.append((action, reasoning))
    return actions


def aggregate_metrics(all_results: List[ChannelTestResults]) -> Dict[str, Any]:
    all_comps: List[ComparisonResult] = []
    for ch_result in all_results:
        all_comps.extend(ch_result.comparisons)

    if not all_comps:
        return {
            "split_recall": 0.0,
            "discard_recall": 0.0,
            "merge_recall": 0.0,
            "overall_accuracy": 0.0,
            "n_split": 0,
            "n_discard": 0,
            "n_merge": 0,
            "n_total": 0,
            "split_correct": 0,
            "discard_correct": 0,
            "merge_correct": 0,
            "total_correct": 0,
        }

    split_comps = [c for c in all_comps if c.action_type == "split"]
    discard_comps = [c for c in all_comps if c.action_type == "discard"]
    merge_comps = [c for c in all_comps if c.action_type == "merge"]

    split_correct = sum(1 for c in split_comps if c.match)
    discard_correct = sum(1 for c in discard_comps if c.match)
    merge_correct = sum(1 for c in merge_comps if c.match)
    total_correct = sum(1 for c in all_comps if c.match)

    return {
        "split_recall": split_correct / len(split_comps) if split_comps else 0.0,
        "discard_recall": discard_correct / len(discard_comps) if discard_comps else 0.0,
        "merge_recall": merge_correct / len(merge_comps) if merge_comps else 0.0,
        "overall_accuracy": total_correct / len(all_comps),
        "n_split": len(split_comps),
        "n_discard": len(discard_comps),
        "n_merge": len(merge_comps),
        "n_total": len(all_comps),
        "split_correct": split_correct,
        "discard_correct": discard_correct,
        "merge_correct": merge_correct,
        "total_correct": total_correct,
    }


def _save_mode_outputs(
    *,
    output_base: Path,
    mode_name: str,
    all_results: List[ChannelTestResults],
    temp_smoke: bool,
    rag_memory_path: Optional[Path] = None,
) -> Dict[str, Any]:
    all_comps: List[ComparisonResult] = []
    for ch in all_results:
        all_comps.extend(ch.comparisons)

    rows = [
        {
            "channel": c.channel,
            "step": c.step,
            "action_type": c.action_type,
            "cluster_id": c.cluster_id,
            "merge_target_id": c.merge_target_id,
            "human_decision": c.human_decision,
            "vlm_decision": c.vlm_decision,
            "match": c.match,
            "human_reasoning": c.human_reasoning,
            "vlm_reasoning": c.vlm_reasoning,
        }
        for c in all_comps
    ]

    detail_path = output_base / f"detailed_comparisons_{mode_name}.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)

    summary = {
        "mode": mode_name,
        "temp_smoke": temp_smoke,
        "overall": aggregate_metrics(all_results),
        "per_channel": {
            ch.channel: {
                **ch.get_metrics(),
                "temp_smoke_truncated": ch.temp_smoke_truncated,
            }
            for ch in all_results
        },
    }
    if rag_memory_path is not None:
        summary["rag_memory_path"] = str(rag_memory_path)

    summary_path = output_base / f"summary_{mode_name}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {mode_name} details to {detail_path}")
    print(f"✓ Saved {mode_name} summary to {summary_path}")
    return summary


def _print_mode_summary(mode_name: str, summary: Dict[str, Any]) -> None:
    overall = summary["overall"]
    tag = " [TEMP SMOKE]" if summary.get("temp_smoke") else ""
    print("\n" + "=" * 80)
    print(f"SUMMARY [{mode_name.upper()}]{tag}")
    print("=" * 80)
    print(f"Total comparisons: {overall['n_total']}")
    print(
        f"Overall accuracy: {overall['overall_accuracy']:.1%} "
        f"({overall['total_correct']}/{overall['n_total']})"
    )
    print(
        f"Split recall: {overall['split_recall']:.1%} ({overall['split_correct']}/{overall['n_split']})"
    )
    print(
        f"Discard recall: {overall['discard_recall']:.1%} ({overall['discard_correct']}/{overall['n_discard']})"
    )
    print(
        f"Merge recall: {overall['merge_recall']:.1%} ({overall['merge_correct']}/{overall['n_merge']})"
    )


def _create_mode_compare(
    no_rag_summary: Dict[str, Any],
    rag_summary: Dict[str, Any],
) -> Dict[str, Any]:
    compare = {
        "temp_smoke": bool(no_rag_summary.get("temp_smoke") or rag_summary.get("temp_smoke")),
        "no_rag": no_rag_summary,
        "rag": rag_summary,
        "delta": {
            "overall_accuracy": rag_summary["overall"]["overall_accuracy"]
            - no_rag_summary["overall"]["overall_accuracy"],
            "split_recall": rag_summary["overall"]["split_recall"]
            - no_rag_summary["overall"]["split_recall"],
            "discard_recall": rag_summary["overall"]["discard_recall"]
            - no_rag_summary["overall"]["discard_recall"],
            "merge_recall": rag_summary["overall"]["merge_recall"]
            - no_rag_summary["overall"]["merge_recall"],
        },
        "per_channel_delta": {},
    }

    all_channels = sorted(
        set(no_rag_summary.get("per_channel", {}).keys())
        | set(rag_summary.get("per_channel", {}).keys())
    )
    for ch in all_channels:
        no_rag_ch = no_rag_summary.get("per_channel", {}).get(ch, {})
        rag_ch = rag_summary.get("per_channel", {}).get(ch, {})
        compare["per_channel_delta"][ch] = {
            "overall_accuracy": rag_ch.get("overall_accuracy", 0.0)
            - no_rag_ch.get("overall_accuracy", 0.0),
            "split_recall": rag_ch.get("split_recall", 0.0)
            - no_rag_ch.get("split_recall", 0.0),
            "discard_recall": rag_ch.get("discard_recall", 0.0)
            - no_rag_ch.get("discard_recall", 0.0),
            "merge_recall": rag_ch.get("merge_recall", 0.0)
            - no_rag_ch.get("merge_recall", 0.0),
        }

    return compare


def test_channel(
    *,
    channel: str,
    mode_name: str,
    output_base: Path,
    auto_discard_threshold: int,
    provider: str,
    model: str,
    use_mock: bool,
    temperature: float,
    reasoning_effort: Optional[str],
    rag_memory: Optional[ContinualRAGMemory],
    rag_top_k: int,
    max_steps_per_channel: Optional[int],
) -> ChannelTestResults:
    print(f"\n{'=' * 80}")
    print(f"TESTING CHANNEL: {channel} [{mode_name}]")
    print(f"{'=' * 80}")

    results = ChannelTestResults(channel=channel)

    print(f"\n[1] Loading data for {channel}...")
    mat_file = f"data/{channel}_spikes.mat"
    data = load_matlab_spikes(mat_file)

    waveforms = data["waveforms"]
    spike_times = data["spiketimes"]
    Fs = data["Fs"]
    hierarchy_tree = data["hierarchy_tree"]
    overcluster_assigns = data["overcluster_assigns"]
    hierarchy_assigns = data["hierarchy_assigns"]

    print(f"  ✓ Loaded {len(spike_times)} spikes, Fs={Fs} Hz")

    print("\n[2] Initializing cluster manager...")
    manager = ClusterManager(
        initial_assigns=hierarchy_assigns,
        overcluster_assigns=overcluster_assigns,
        hierarchy_tree=hierarchy_tree,
        spike_times=spike_times,
        waveforms=waveforms,
    )
    print(f"  ✓ Initial clusters: {len(manager.get_active_clusters())}")

    print("\n[3] Computing cluster features...")
    _ = ClusterFeatures(meta=data, assigns=manager.assigns)
    print("  ✓ Features ready")

    print(f"\n[4] Phase 0: Auto-discard (<{auto_discard_threshold} spikes)...")
    _, discarded_actions = automatic_size_filter(manager.assigns, auto_discard_threshold)
    for action in discarded_actions:
        manager.discard_cluster(action.cluster_id)
    print(f"  ✓ Auto-discarded {len(discarded_actions)} clusters")

    print("\n[5] Loading human action sheet...")
    action_sheet = load_action_sheet(channel)
    print(f"  ✓ Loaded {len(action_sheet)} human actions")

    print("\n[6] Replaying human actions and comparing with VLM...")

    for idx, (action_str, human_reasoning) in enumerate(action_sheet, start=1):
        if max_steps_per_channel is not None and idx > max_steps_per_channel:
            results.temp_smoke_truncated = True
            print(
                f"  [TEMP SMOKE] Step limit reached for {channel}: "
                f"{max_steps_per_channel}"
            )
            break

        try:
            action_type, cluster_id, target_id = parse_action(action_str)
        except ValueError as exc:
            print(f"  ⚠ Step {idx}: {exc} - skipping")
            continue

        print(f"\n  Step {idx}/{len(action_sheet)}: {action_str}")

        info = manager.get_cluster_info(cluster_id)
        if info is None or info["n_spikes"] == 0:
            print(f"    ⚠ Cluster {cluster_id} missing/already discarded - skipping")
            continue

        step_out = output_base / mode_name / channel / f"step_{idx:03d}"
        retrieved_examples: Optional[List[Dict[str, Any]]] = None
        vlm_response: Dict[str, Any]

        if action_type in ["split", "discard"]:
            if rag_memory is not None:
                retrieved_examples = rag_memory.retrieve_phase1(
                    waveforms=info["waveforms"],
                    spike_times=info["spike_times"],
                    n_spikes=int(info["n_spikes"]),
                    top_k=rag_top_k,
                )
                print(f"    [RAG] Retrieved {len(retrieved_examples)} phase1 examples")

            vlm_response = vlm_phase1_cluster_decision(
                cluster_id=cluster_id,
                waveforms=info["waveforms"],
                spike_times=info["spike_times"],
                overcluster_composition=info["overclusters"],
                hierarchy_tree=manager.hierarchy_tree,
                sampling_rate=Fs,
                provider=provider,
                model=model,
                use_mock=use_mock,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                output_dir=step_out,
                retrieved_examples=retrieved_examples,
            )
            expected = "SPLIT" if action_type == "split" else "DISCARD"
            vlm_decision = str(vlm_response.get("action", ""))

            comp = ComparisonResult(
                channel=channel,
                step=idx,
                action_type=action_type,
                cluster_id=cluster_id,
                merge_target_id=None,
                human_decision=expected,
                vlm_decision=vlm_decision,
                match=(vlm_decision == expected),
                human_reasoning=human_reasoning,
                vlm_reasoning=str(vlm_response.get("rationale", "")),
            )
            results.add_comparison(comp)

            if rag_memory is not None:
                rag_entry = build_phase1_memory_entry(
                    channel_id=channel,
                    step=idx,
                    cluster_id=cluster_id,
                    waveforms=info["waveforms"],
                    spike_times=info["spike_times"],
                    gt_action=expected,
                    gt_reasoning=human_reasoning if human_reasoning else None,
                    prompt_text=str(vlm_response.get("prompt_text", "")),
                    image_paths=[str(p) for p in vlm_response.get("image_paths", [])],
                )
                rag_memory.add(rag_entry)

        else:
            large_info = manager.get_cluster_info(int(target_id)) if target_id is not None else None
            if large_info is None or large_info["n_spikes"] == 0:
                print(f"    ⚠ Merge target {target_id} missing/already discarded - skipping")
                continue

            if rag_memory is not None:
                retrieved_examples = rag_memory.retrieve_phase2(
                    small_waveforms=info["waveforms"],
                    small_spike_times=info["spike_times"],
                    large_waveforms=large_info["waveforms"],
                    large_spike_times=large_info["spike_times"],
                    top_k=rag_top_k,
                )
                print(f"    [RAG] Retrieved {len(retrieved_examples)} phase2 examples")

            vlm_response = vlm_phase2_merge_decision(
                small_cluster_id=cluster_id,
                small_waveforms=info["waveforms"],
                small_spike_times=info["spike_times"],
                large_cluster_id=int(target_id),
                large_waveforms=large_info["waveforms"],
                large_spike_times=large_info["spike_times"],
                sampling_rate=Fs,
                provider=provider,
                model=model,
                use_mock=use_mock,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                output_dir=step_out,
                retrieved_examples=retrieved_examples,
            )

            expected = "MERGE"
            vlm_decision = str(vlm_response.get("action", ""))
            comp = ComparisonResult(
                channel=channel,
                step=idx,
                action_type="merge",
                cluster_id=cluster_id,
                merge_target_id=int(target_id),
                human_decision=expected,
                vlm_decision=vlm_decision,
                match=(vlm_decision == expected),
                human_reasoning=human_reasoning,
                vlm_reasoning=str(vlm_response.get("rationale", "")),
            )
            results.add_comparison(comp)

            if rag_memory is not None:
                rag_entry = build_phase2_memory_entry(
                    channel_id=channel,
                    step=idx,
                    cluster_id=cluster_id,
                    target_id=int(target_id),
                    small_waveforms=info["waveforms"],
                    small_spike_times=info["spike_times"],
                    large_waveforms=large_info["waveforms"],
                    large_spike_times=large_info["spike_times"],
                    gt_action=expected,
                    gt_reasoning=human_reasoning if human_reasoning else None,
                    prompt_text=str(vlm_response.get("prompt_text", "")),
                    image_paths=[str(p) for p in vlm_response.get("image_paths", [])],
                )
                rag_memory.add(rag_entry)

        marker = "✓ MATCH" if results.comparisons[-1].match else "✗ MISMATCH"
        print(
            f"    {marker}: Human={results.comparisons[-1].human_decision}, "
            f"VLM={results.comparisons[-1].vlm_decision}"
        )

        # Execute human action (not VLM action) to maintain replay state.
        if action_type == "split":
            try:
                new_ids = manager.split_last_merge(cluster_id)
                print(f"    → Executed human split {cluster_id} -> {new_ids}")
            except Exception as exc:
                print(f"    ✗ Human split failed: {exc}")
        elif action_type == "discard":
            manager.discard_cluster(cluster_id)
            print(f"    → Executed human discard {cluster_id}")
        else:
            manager.merge_clusters([cluster_id, int(target_id)], target_id=int(target_id))
            print(f"    → Executed human merge {cluster_id} -> {target_id}")

    print(
        f"\n✓ {channel} [{mode_name}] complete: "
        f"{len(results.comparisons)} comparisons"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unit test: VLM decision accuracy vs human GT (with optional continual RAG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--auto-discard-threshold", type=int, default=500)
    parser.add_argument("--provider", default="gpt4o")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--output-base", default="output/unit_test_results")
    parser.add_argument("--run-mode", choices=["no_rag", "rag", "both"], default="both")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-waveform-weight", type=float, default=0.7)
    parser.add_argument("--rag-feature-weight", type=float, default=0.3)
    parser.add_argument("--rag-memory-path", default=None)
    parser.add_argument(
        "--rag-persist-memory",
        action="store_true",
        help="Persist RAG memory to JSONL. Default is in-memory only.",
    )
    parser.add_argument(
        "--rag-overwrite-memory",
        action="store_true",
        help="If persisting memory, clear memory file before rag run.",
    )
    parser.add_argument("--max-steps-per-channel", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    reasoning_effort: Optional[str]
    if str(args.reasoning_effort).lower() in {"none", "", "null"}:
        reasoning_effort = None
    else:
        reasoning_effort = str(args.reasoning_effort)

    temp_smoke = args.max_steps_per_channel is not None
    if temp_smoke:
        print(
            f"[TEMP SMOKE] max_steps_per_channel={args.max_steps_per_channel}. "
            "Metrics are partial."
        )

    run_modes = ["no_rag", "rag"] if args.run_mode == "both" else [args.run_mode]

    print("=" * 80)
    print("UNIT TEST: VLM DECISION ACCURACY vs HUMAN GT")
    print("=" * 80)
    print(f"Model: {args.provider}/{args.model}")
    print(f"Channels (order): {', '.join(args.channels)}")
    print(f"Run mode: {args.run_mode}")
    print(f"Use mock: {args.use_mock}")
    print(f"Output: {output_base}")
    if temp_smoke:
        print("[TEMP SMOKE] enabled")
    print("=" * 80)

    mode_summaries: Dict[str, Dict[str, Any]] = {}

    for mode_name in run_modes:
        rag_memory: Optional[ContinualRAGMemory] = None
        rag_memory_path: Optional[Path] = None

        if mode_name == "rag":
            persist_memory = bool(args.rag_persist_memory or args.rag_memory_path)
            if persist_memory:
                rag_memory_path = (
                    Path(args.rag_memory_path)
                    if args.rag_memory_path
                    else output_base / "rag_memory.jsonl"
                )
            rag_memory = ContinualRAGMemory(
                memory_path=rag_memory_path,
                waveform_weight=args.rag_waveform_weight,
                feature_weight=args.rag_feature_weight,
                default_top_k=args.rag_top_k,
            )
            if rag_memory_path is None:
                print("\n[RAG] memory mode: in-memory only (non-persistent)")
            else:
                print(f"\n[RAG] memory path: {rag_memory_path}")
            print(
                f"[RAG] weights: waveform={args.rag_waveform_weight}, "
                f"feature={args.rag_feature_weight}, top_k={args.rag_top_k}"
            )
            print(f"[RAG] overwrite_memory: {args.rag_overwrite_memory}")
            if args.rag_overwrite_memory:
                rag_memory.clear(persist=(rag_memory_path is not None))

        all_results: List[ChannelTestResults] = []
        for channel in args.channels:
            try:
                ch_result = test_channel(
                    channel=channel,
                    mode_name=mode_name,
                    output_base=output_base,
                    auto_discard_threshold=args.auto_discard_threshold,
                    provider=args.provider,
                    model=args.model,
                    use_mock=args.use_mock,
                    temperature=args.temperature,
                    reasoning_effort=reasoning_effort,
                    rag_memory=rag_memory,
                    rag_top_k=args.rag_top_k,
                    max_steps_per_channel=args.max_steps_per_channel,
                )
                all_results.append(ch_result)
            except Exception as exc:
                print(f"\n✗ {channel} [{mode_name}] failed: {exc}")
                import traceback

                traceback.print_exc()

        summary = _save_mode_outputs(
            output_base=output_base,
            mode_name=mode_name,
            all_results=all_results,
            temp_smoke=temp_smoke,
            rag_memory_path=rag_memory_path,
        )
        _print_mode_summary(mode_name, summary)
        mode_summaries[mode_name] = summary

    if "no_rag" in mode_summaries and "rag" in mode_summaries:
        compare = _create_mode_compare(
            no_rag_summary=mode_summaries["no_rag"],
            rag_summary=mode_summaries["rag"],
        )
        compare_path = output_base / "summary_compare.json"
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(compare, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print("COMPARE [RAG - NO_RAG]")
        print("=" * 80)
        print(f"Overall accuracy delta: {compare['delta']['overall_accuracy']:+.4f}")
        print(f"Split recall delta: {compare['delta']['split_recall']:+.4f}")
        print(f"Discard recall delta: {compare['delta']['discard_recall']:+.4f}")
        print(f"Merge recall delta: {compare['delta']['merge_recall']:+.4f}")
        print(f"✓ Saved compare summary to {compare_path}")

    print("\n✓ Unit testing complete!")


if __name__ == "__main__":
    main()
