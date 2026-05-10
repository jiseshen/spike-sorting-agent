"""
Run full-channel simulated teacher feedback and generate token/cost budget report.

Workflow:
1) Load one channel and apply phase-0 auto-discard for small clusters.
2) Replay action sheet step-by-step to match human state progression.
3) At each valid step:
   - call student model (phase1 or phase2)
   - call teacher model with student observation context + student output
4) Collect per-call token usage and estimate cost from model pricing.
5) Export per-step details and aggregated/projection budget summary.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agent.api import get_last_call_meta, reset_call_tracking
from src.agent.runner import (
    call_vlm_api,
    compute_merged_isi_violation_rate,
    compute_waveform_correlation,
    create_aggregation_tree_image,
    create_isi_histogram_image,
    create_waveform_overlay_image,
    vlm_phase1_cluster_decision,
    vlm_phase2_merge_decision,
)
from src.agent.teacher_feedback import build_teacher_feedback_prompt
from src.cluster.auto_filter import automatic_size_filter
from src.cluster.manager import ClusterManager
from src.io.matlab_loader import load_matlab_spikes


PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.4},
    "gpt-4o": {"input": 2.5, "cached_input": 1.25, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.6},
}

PRICING_SOURCE = "https://platform.openai.com/docs/pricing"
PRICING_NOTE = "Text token prices per 1M tokens, standard pricing."


@dataclass
class RoleUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def parse_action(action_str: str) -> Tuple[str, int, Optional[int]]:
    action_str = action_str.strip().strip("'\"")
    split_match = re.match(r"s\s+(\d+)", action_str)
    if split_match:
        return "split", int(split_match.group(1)), None

    merge_match = re.match(r"m\s+(\d+)\s+(\d+)", action_str)
    if merge_match:
        target = int(merge_match.group(1))
        source = int(merge_match.group(2))
        if target == 0:
            return "discard", source, None
        return "merge", source, target

    raise ValueError(f"Cannot parse action: {action_str}")


def load_action_sheet(channel: str) -> List[Tuple[str, str]]:
    csv_path = Path(f"data/action_sheets/{channel}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Action sheet not found: {csv_path}")
    actions: List[Tuple[str, str]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append((row["Actions"].strip(), row.get("Action Reasoning", "").strip()))
    return actions


def resolve_pricing_key(model: str) -> Optional[str]:
    model_lower = model.lower()
    for key in sorted(PRICING_USD_PER_1M.keys(), key=len, reverse=True):
        if model_lower == key or model_lower.startswith(f"{key}-"):
            return key
    return None


def usage_from_meta(meta: Dict[str, Any]) -> RoleUsage:
    usage = meta.get("usage", {}) if isinstance(meta, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    return RoleUsage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        cached_input_tokens=int(usage.get("cached_input_tokens", 0) or 0),
        uncached_input_tokens=int(usage.get("uncached_input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        reasoning_output_tokens=int(usage.get("reasoning_output_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
    )


def cost_for_usage(usage: RoleUsage, pricing: Dict[str, float]) -> float:
    return (
        usage.uncached_input_tokens * pricing["input"]
        + usage.cached_input_tokens * pricing["cached_input"]
        + usage.output_tokens * pricing["output"]
    ) / 1_000_000.0


def setup_channel_manager(channel: str, auto_discard_threshold: int) -> Tuple[ClusterManager, float]:
    data = load_matlab_spikes(f"data/{channel}_spikes.mat")
    manager = ClusterManager(
        initial_assigns=data["hierarchy_assigns"],
        overcluster_assigns=data["overcluster_assigns"],
        hierarchy_tree=data["hierarchy_tree"],
        spike_times=data["spiketimes"],
        waveforms=data["waveforms"],
    )
    _, discarded_actions = automatic_size_filter(manager.assigns, auto_discard_threshold)
    for action in discarded_actions:
        manager.discard_cluster(action.cluster_id)
    return manager, data["Fs"]


def execute_human_action(
    manager: ClusterManager,
    action_type: str,
    cluster_id: int,
    target_id: Optional[int],
) -> None:
    if action_type == "split":
        try:
            manager.split_last_merge(cluster_id)
        except Exception:
            pass
    elif action_type == "discard":
        manager.discard_cluster(cluster_id)
    elif action_type == "merge" and target_id is not None:
        manager.merge_clusters([cluster_id, target_id], target_id=target_id)


def aggregate_role(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    keys = [
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "cost_usd",
    ]
    calls = len(rows)
    missing_usage_calls = sum(
        1
        for r in rows
        if (r.get(f"{prefix}_input_tokens", 0) == 0 and r.get(f"{prefix}_output_tokens", 0) == 0)
    )
    totals = {
        k: sum(float(r.get(f"{prefix}_{k}", 0) or 0) for r in rows)
        for k in keys
    }
    avgs = {f"avg_{k}": (totals[k] / calls if calls else 0.0) for k in keys}
    cost_values = [float(r.get(f"{prefix}_cost_usd", 0) or 0) for r in rows]
    if cost_values:
        sorted_costs = sorted(cost_values)
        p50_idx = int(0.50 * (len(sorted_costs) - 1))
        p95_idx = int(0.95 * (len(sorted_costs) - 1))
        p50_cost = sorted_costs[p50_idx]
        p95_cost = sorted_costs[p95_idx]
        max_cost = sorted_costs[-1]
    else:
        p50_cost = 0.0
        p95_cost = 0.0
        max_cost = 0.0
    return {
        "calls": calls,
        "missing_usage_calls": missing_usage_calls,
        **totals,
        **avgs,
        "p50_call_cost_usd": p50_cost,
        "p95_call_cost_usd": p95_cost,
        "max_call_cost_usd": max_cost,
    }


def run_channel_simulation(
    *,
    channel: str,
    provider: str,
    student_model: str,
    teacher_model: str,
    reasoning_effort: str,
    temperature: float,
    use_mock: bool,
    auto_discard_threshold: int,
    output_base: Path,
    projection_channels: int,
    projection_steps_per_channel: int,
) -> Dict[str, Any]:
    if not Path(f"data/{channel}_spikes.mat").exists():
        raise FileNotFoundError(f"Missing channel data: data/{channel}_spikes.mat")

    action_sheet = load_action_sheet(channel)
    manager, Fs = setup_channel_manager(channel, auto_discard_threshold)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_base / "runs" / f"{channel}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    student_price_key = resolve_pricing_key(student_model)
    teacher_price_key = resolve_pricing_key(teacher_model)
    if student_price_key is None:
        raise ValueError(f"Unsupported student model for pricing: {student_model}")
    if teacher_price_key is None:
        raise ValueError(f"Unsupported teacher model for pricing: {teacher_model}")
    student_pricing = PRICING_USD_PER_1M[student_price_key]
    teacher_pricing = PRICING_USD_PER_1M[teacher_price_key]

    reset_call_tracking()
    step_rows: List[Dict[str, Any]] = []

    for step, (action_str, human_reasoning) in enumerate(action_sheet, start=1):
        row: Dict[str, Any] = {
            "channel": channel,
            "step": step,
            "action_str": action_str,
            "human_reasoning": human_reasoning,
            "status": "processed",
        }

        try:
            action_type, cluster_id, target_id = parse_action(action_str)
        except ValueError as exc:
            row["status"] = "skipped_parse_error"
            row["error"] = str(exc)
            step_rows.append(row)
            continue

        row["action_type"] = action_type
        row["cluster_id"] = cluster_id
        row["target_id"] = target_id
        expected_human_action = {"split": "SPLIT", "discard": "DISCARD", "merge": "MERGE"}[action_type]
        row["human_action"] = expected_human_action

        info = manager.get_cluster_info(cluster_id)
        if info is None or info["n_spikes"] == 0:
            row["status"] = "skipped_missing_cluster"
            step_rows.append(row)
            continue

        step_dir = run_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if action_type in {"split", "discard"}:
            student_resp = vlm_phase1_cluster_decision(
                cluster_id=cluster_id,
                waveforms=info["waveforms"],
                spike_times=info["spike_times"],
                overcluster_composition=info["overclusters"],
                hierarchy_tree=manager.hierarchy_tree,
                sampling_rate=Fs,
                provider=provider,
                model=student_model,
                use_mock=use_mock,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                output_dir=step_dir / "student",
            )
            teacher_images = [
                create_waveform_overlay_image(info["waveforms"], cluster_id, Fs),
                create_isi_histogram_image(info["spike_times"], cluster_id),
                create_aggregation_tree_image(manager.hierarchy_tree, info["overclusters"], cluster_id),
            ]
            teacher_context = {
                "cluster_id": cluster_id,
                "n_spikes": int(info["n_spikes"]),
                "n_overclusters": len(info["overclusters"]),
            }
        else:
            target_info = manager.get_cluster_info(target_id)
            if target_info is None or target_info["n_spikes"] == 0:
                row["status"] = "skipped_missing_target"
                step_rows.append(row)
                continue
            student_resp = vlm_phase2_merge_decision(
                small_cluster_id=cluster_id,
                small_waveforms=info["waveforms"],
                small_spike_times=info["spike_times"],
                large_cluster_id=target_id,
                large_waveforms=target_info["waveforms"],
                large_spike_times=target_info["spike_times"],
                sampling_rate=Fs,
                provider=provider,
                model=student_model,
                use_mock=use_mock,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                output_dir=step_dir / "student",
            )
            corr = compute_waveform_correlation(info["waveforms"], target_info["waveforms"])
            merged_isi = compute_merged_isi_violation_rate(
                info["spike_times"], target_info["spike_times"]
            )
            small_isi = compute_merged_isi_violation_rate(info["spike_times"], info["spike_times"][:1])
            large_isi = compute_merged_isi_violation_rate(
                target_info["spike_times"], target_info["spike_times"][:1]
            )
            teacher_images = [
                create_waveform_overlay_image(info["waveforms"], cluster_id, Fs),
                create_waveform_overlay_image(target_info["waveforms"], target_id, Fs),
                create_isi_histogram_image(
                    np.sort(np.concatenate([info["spike_times"], target_info["spike_times"]])),
                    f"{cluster_id}+{target_id}",
                ),
            ]
            teacher_context = {
                "cluster_id": cluster_id,
                "merge_target_id": int(target_id),
                "n_small": int(info["n_spikes"]),
                "n_large": int(target_info["n_spikes"]),
                "small_isi_rate": float(small_isi),
                "large_isi_rate": float(large_isi),
                "waveform_correlation": float(corr),
                "merged_isi_rate": float(merged_isi),
            }

        student_meta = get_last_call_meta()
        student_usage = usage_from_meta(student_meta)
        student_usage.cost_usd = cost_for_usage(student_usage, student_pricing)

        student_action = student_resp.get("action", "")
        student_rationale = student_resp.get("rationale", "")
        student_match = int(student_action == expected_human_action)

        teacher_prompt = build_teacher_feedback_prompt(
            channel=channel,
            step=step,
            action_type=action_type,
            context_summary=teacher_context,
            student_action=student_action,
            student_rationale=student_rationale,
            human_action=expected_human_action,
            human_reasoning=human_reasoning,
            merge_target_id=target_id,
        )
        teacher_feedback = call_vlm_api(
            prompt=teacher_prompt,
            images=teacher_images,
            model=teacher_model,
            provider=provider,
            use_mock=use_mock,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        teacher_meta = get_last_call_meta()
        teacher_usage = usage_from_meta(teacher_meta)
        teacher_usage.cost_usd = cost_for_usage(teacher_usage, teacher_pricing)

        (step_dir / "teacher_prompt.txt").write_text(teacher_prompt)
        (step_dir / "teacher_feedback.txt").write_text(teacher_feedback)
        (step_dir / "student_response.json").write_text(
            json.dumps(student_resp, ensure_ascii=False, indent=2)
        )

        row.update(
            {
                "student_action": student_action,
                "student_rationale": student_rationale,
                "student_match": student_match,
                "teacher_feedback": teacher_feedback.strip(),
                "student_model_requested": student_model,
                "student_model_actual": student_meta.get("actual_model"),
                "teacher_model_requested": teacher_model,
                "teacher_model_actual": teacher_meta.get("actual_model"),
                "student_input_tokens": student_usage.input_tokens,
                "student_cached_input_tokens": student_usage.cached_input_tokens,
                "student_uncached_input_tokens": student_usage.uncached_input_tokens,
                "student_output_tokens": student_usage.output_tokens,
                "student_reasoning_output_tokens": student_usage.reasoning_output_tokens,
                "student_total_tokens": student_usage.total_tokens,
                "student_cost_usd": student_usage.cost_usd,
                "teacher_input_tokens": teacher_usage.input_tokens,
                "teacher_cached_input_tokens": teacher_usage.cached_input_tokens,
                "teacher_uncached_input_tokens": teacher_usage.uncached_input_tokens,
                "teacher_output_tokens": teacher_usage.output_tokens,
                "teacher_reasoning_output_tokens": teacher_usage.reasoning_output_tokens,
                "teacher_total_tokens": teacher_usage.total_tokens,
                "teacher_cost_usd": teacher_usage.cost_usd,
                "step_cost_usd": student_usage.cost_usd + teacher_usage.cost_usd,
            }
        )
        step_rows.append(row)

        execute_human_action(manager, action_type, cluster_id, target_id)

    step_rows_csv = run_dir / "step_call_stats.csv"
    fieldnames: List[str] = []
    for item in step_rows:
        for key in item.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(step_rows_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    processed_rows = [r for r in step_rows if r.get("status") == "processed"]
    student_summary = aggregate_role(processed_rows, "student")
    teacher_summary = aggregate_role(processed_rows, "teacher")
    total_cost = student_summary["cost_usd"] + teacher_summary["cost_usd"]
    total_calls = student_summary["calls"] + teacher_summary["calls"]
    total_processed_steps = len(processed_rows)
    accuracy = (
        sum(int(r.get("student_match", 0)) for r in processed_rows) / total_processed_steps
        if total_processed_steps
        else 0.0
    )

    avg_student_cost_per_step = (
        student_summary["cost_usd"] / total_processed_steps if total_processed_steps else 0.0
    )
    avg_teacher_cost_per_step = (
        teacher_summary["cost_usd"] / total_processed_steps if total_processed_steps else 0.0
    )
    projected_steps = projection_channels * projection_steps_per_channel

    projection = {
        "projection_channels": projection_channels,
        "projection_steps_per_channel": projection_steps_per_channel,
        "projected_steps_total": projected_steps,
        "student_projected_cost_usd": avg_student_cost_per_step * projected_steps,
        "teacher_projected_cost_usd": avg_teacher_cost_per_step * projected_steps,
        "total_projected_cost_usd": (avg_student_cost_per_step + avg_teacher_cost_per_step)
        * projected_steps,
        "student_projected_input_tokens": student_summary["avg_input_tokens"] * projected_steps,
        "student_projected_output_tokens": student_summary["avg_output_tokens"] * projected_steps,
        "teacher_projected_input_tokens": teacher_summary["avg_input_tokens"] * projected_steps,
        "teacher_projected_output_tokens": teacher_summary["avg_output_tokens"] * projected_steps,
    }

    summary = {
        "run_id": run_id,
        "channel": channel,
        "models": {
            "provider": provider,
            "student_model_requested": student_model,
            "teacher_model_requested": teacher_model,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "use_mock": use_mock,
        },
        "pricing": {
            "source": PRICING_SOURCE,
            "note": PRICING_NOTE,
            "student_pricing_key": student_price_key,
            "teacher_pricing_key": teacher_price_key,
            "student_usd_per_1m": student_pricing,
            "teacher_usd_per_1m": teacher_pricing,
        },
        "counts": {
            "action_sheet_steps": len(action_sheet),
            "processed_steps": total_processed_steps,
            "skipped_steps": len(action_sheet) - total_processed_steps,
            "total_model_calls": total_calls,
        },
        "quality": {
            "student_action_match_rate": accuracy,
        },
        "student_summary": student_summary,
        "teacher_summary": teacher_summary,
        "combined": {
            "total_cost_usd": total_cost,
            "avg_cost_per_model_call_usd": (total_cost / total_calls if total_calls else 0.0),
            "avg_cost_per_processed_step_usd": (
                total_cost / total_processed_steps if total_processed_steps else 0.0
            ),
        },
        "projection_100x50": projection,
        "artifacts": {
            "run_dir": str(run_dir),
            "step_call_stats_csv": str(step_rows_csv),
        },
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_base / "latest_budget.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Channel Budget Report",
        "",
        f"- channel: {channel}",
        f"- run_id: {run_id}",
        f"- student model: {student_model} (pricing key: {student_price_key})",
        f"- teacher model: {teacher_model} (pricing key: {teacher_price_key})",
        f"- processed steps: {total_processed_steps}/{len(action_sheet)}",
        f"- student action match rate: {accuracy:.2%}",
        "",
        "## Costs",
        f"- student total cost: ${student_summary['cost_usd']:.6f}",
        f"- teacher total cost: ${teacher_summary['cost_usd']:.6f}",
        f"- combined total cost: ${total_cost:.6f}",
        f"- avg cost per model call: ${summary['combined']['avg_cost_per_model_call_usd']:.6f}",
        f"- avg cost per processed step: ${summary['combined']['avg_cost_per_processed_step_usd']:.6f}",
        "",
        "## 100 Channels x 50 Steps Projection",
        f"- projected steps: {projected_steps}",
        f"- student projected cost: ${projection['student_projected_cost_usd']:.2f}",
        f"- teacher projected cost: ${projection['teacher_projected_cost_usd']:.2f}",
        f"- total projected cost: ${projection['total_projected_cost_usd']:.2f}",
        "",
        "## Pricing Source",
        f"- {PRICING_SOURCE}",
    ]
    (run_dir / "report.md").write_text("\n".join(lines))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-channel teacher budget simulation.")
    parser.add_argument("--channel", type=str, default="CH20")
    parser.add_argument("--provider", type=str, default="gpt4o")
    parser.add_argument("--student-model", type=str, default="gpt-5-mini")
    parser.add_argument("--teacher-model", type=str, default="gpt-5")
    parser.add_argument("--reasoning-effort", type=str, default="minimal")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--auto-discard-threshold", type=int, default=500)
    parser.add_argument("--projection-channels", type=int, default=100)
    parser.add_argument("--projection-steps-per-channel", type=int, default=50)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("output/simulated_teacher_budget"),
    )
    args = parser.parse_args()

    summary = run_channel_simulation(
        channel=args.channel,
        provider=args.provider,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        use_mock=args.use_mock,
        auto_discard_threshold=args.auto_discard_threshold,
        output_base=args.output_base,
        projection_channels=args.projection_channels,
        projection_steps_per_channel=args.projection_steps_per_channel,
    )

    print("\n=== Channel Teacher Budget Simulation ===")
    print(f"channel={summary['channel']} run_id={summary['run_id']}")
    print(
        "steps processed="
        f"{summary['counts']['processed_steps']}/{summary['counts']['action_sheet_steps']}"
    )
    print(f"student cost=${summary['student_summary']['cost_usd']:.6f}")
    print(f"teacher cost=${summary['teacher_summary']['cost_usd']:.6f}")
    print(f"total cost=${summary['combined']['total_cost_usd']:.6f}")
    print(f"projected total (100x50)=${summary['projection_100x50']['total_projected_cost_usd']:.2f}")
    print(f"latest summary: output/simulated_teacher_budget/latest_budget.json")


if __name__ == "__main__":
    main()
