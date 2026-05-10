"""
Alignment report generator.

Aggregates per-channel ActionMetrics and reasoning alignment scores into
a single alignment_report.json per channel, and an optional setting-level
summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .action_metrics import ActionMetrics, load_predictions_and_compute
from .reasoning_metrics import compute_reasoning_alignment


def generate_alignment_report(
    setting_id: str,
    channel_id: str,
    output_dir: str | Path,
    reasoning_method: str = "cosine",
    llm_judge_model: Optional[str] = None,
    provider: str = "gpt4o",
    use_mock: bool = False,
    force: bool = False,
) -> dict:
    """
    Generate the alignment_report.json for one channel.

    Reads:
      output/<setting_id>/<channel_id>/eval/eval_predictions.jsonl
      output/<setting_id>/<channel_id>/trajectory/trajectory.jsonl (for rationales)

    Writes:
      output/<setting_id>/<channel_id>/eval/alignment_report.json

    Returns the report dict.
    """
    output_dir = Path(output_dir)
    eval_dir = output_dir / setting_id / channel_id / "eval"
    report_file = eval_dir / "alignment_report.json"

    if report_file.exists() and not force:
        print(f"  [skip] {report_file} exists")
        with open(report_file) as f:
            return json.load(f)

    # --- Action metrics ---
    action_metrics: ActionMetrics = load_predictions_and_compute(
        eval_dir=eval_dir,
        channel_id=channel_id,
        setting_id=setting_id,
    )

    # --- Reasoning alignment (from trajectory.jsonl) ---
    traj_file = output_dir / setting_id / channel_id / "trajectory" / "trajectory.jsonl"
    reasoning_result: dict = {"method": reasoning_method, "mean_score": 0.0, "per_step_scores": []}

    if traj_file.exists():
        from src.trajectories.record import load_trajectory
        steps = load_trajectory(traj_file)
        student_rationales = [s.student_rationale for s in steps]
        teacher_feedbacks = [s.teacher_feedback for s in steps]

        if student_rationales:
            reasoning_result = compute_reasoning_alignment(
                student_rationales=student_rationales,
                teacher_feedbacks=teacher_feedbacks,
                method=reasoning_method,
                llm_judge_model=llm_judge_model,
                provider=provider,
                use_mock=use_mock,
            )

    report = {
        "channel_id": channel_id,
        "setting_id": setting_id,
        "action_accuracy": action_metrics.overall_accuracy,
        "per_type_accuracy": action_metrics.per_type_accuracy,
        "confusion_matrix": action_metrics.confusion_matrix,
        "edit_distance": action_metrics.edit_distance,
        "n_steps": action_metrics.n_steps,
        f"reasoning_{reasoning_method}_sim": reasoning_result["mean_score"],
        "reasoning_alignment": reasoning_result,
    }

    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Write confusion matrix as CSV
    _write_confusion_csv(action_metrics.confusion_matrix, eval_dir / "action_confusion_matrix.csv")

    print(
        f"  [done] {setting_id}/{channel_id}: "
        f"action_acc={report['action_accuracy']:.3f}, "
        f"reasoning_sim={report[f'reasoning_{reasoning_method}_sim']:.3f}"
    )
    return report


def aggregate_setting_reports(
    setting_id: str,
    output_dir: str | Path,
) -> dict:
    """
    Aggregate alignment_report.json files across all channels in a setting.

    Returns a summary dict and writes <setting_id>_alignment_summary.json.
    """
    output_dir = Path(output_dir)
    setting_dir = output_dir / setting_id

    reports: list[dict] = []
    for ch_dir in sorted(setting_dir.iterdir()):
        report_file = ch_dir / "eval" / "alignment_report.json"
        if report_file.exists():
            with open(report_file) as f:
                reports.append(json.load(f))

    if not reports:
        return {}

    def _mean(key: str) -> float:
        vals = [r[key] for r in reports if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    reasoning_key = next(
        (k for k in reports[0] if k.startswith("reasoning_") and k.endswith("_sim")),
        None,
    )

    summary = {
        "setting_id": setting_id,
        "n_channels": len(reports),
        "mean_action_accuracy": _mean("action_accuracy"),
        "mean_edit_distance": _mean("edit_distance"),
        "mean_reasoning_sim": _mean(reasoning_key) if reasoning_key else 0.0,
        "reasoning_key": reasoning_key,
    }

    summary_file = output_dir / f"{setting_id}_alignment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _write_confusion_csv(confusion: dict, path: Path) -> None:
    all_labels = sorted({lbl for row in confusion.values() for lbl in row} | set(confusion.keys()))
    lines = ["gt\\" + "pred," + ",".join(all_labels)]
    for gt in all_labels:
        row = confusion.get(gt, {})
        counts = [str(row.get(pred, 0)) for pred in all_labels]
        lines.append(f"{gt}," + ",".join(counts))
    path.write_text("\n".join(lines))
