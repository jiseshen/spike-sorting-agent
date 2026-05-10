"""
Cross-setting aggregation.

Reads all alignment_report.json files produced by Stage 5 and produces
a single sweep_summary.json with per-setting and overall statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def aggregate_sweep_results(output_dir: str | Path) -> dict:
    """
    Aggregate alignment reports across all settings in output_dir.

    Reads:
      output/<setting_id>/<channel_id>/eval/alignment_report.json

    Writes:
      output/sweep_summary.json

    Returns:
        Aggregated summary dict.
    """
    output_dir = Path(output_dir)

    setting_summaries: list[dict] = []

    for setting_dir in sorted(output_dir.iterdir()):
        if not setting_dir.is_dir():
            continue
        setting_id = setting_dir.name

        # Skip non-setting directories (e.g. "sweep_summary.json" parent)
        channel_dirs = [d for d in setting_dir.iterdir() if d.is_dir() and d.name.startswith("ch_")]
        if not channel_dirs:
            continue

        channel_reports: list[dict] = []
        for ch_dir in sorted(channel_dirs):
            report_file = ch_dir / "eval" / "alignment_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    channel_reports.append(json.load(f))

        if not channel_reports:
            continue

        def _mean(key: str) -> float:
            vals = [r[key] for r in channel_reports if key in r and isinstance(r[key], (int, float))]
            return sum(vals) / len(vals) if vals else 0.0

        reasoning_key = next(
            (k for k in channel_reports[0] if k.startswith("reasoning_") and k.endswith("_sim")),
            None,
        )

        setting_summary = {
            "setting_id": setting_id,
            "n_channels": len(channel_reports),
            "mean_action_accuracy": _mean("action_accuracy"),
            "mean_edit_distance": _mean("edit_distance"),
            "mean_n_steps": _mean("n_steps"),
            "mean_reasoning_sim": _mean(reasoning_key) if reasoning_key else 0.0,
        }

        # Load setting config to capture experimental conditions
        config_file = setting_dir / "setting_config.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    cfg = yaml.safe_load(f) or {}
                setting_summary["teacher_style"] = (
                    cfg.get("teacher_criteria", {}).get("teacher_style", "unknown")
                )
                setting_summary["noise_level"] = cfg.get("noise", {}).get("noise_level")
                setting_summary["drift_enabled"] = cfg.get("drift", {}).get("enabled")
            except Exception:
                pass

        setting_summaries.append(setting_summary)

    if not setting_summaries:
        return {"settings": [], "overall": {}}

    def _overall_mean(key: str) -> float:
        vals = [s[key] for s in setting_summaries if key in s and isinstance(s[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n_settings": len(setting_summaries),
        "settings": setting_summaries,
        "overall": {
            "mean_action_accuracy": _overall_mean("mean_action_accuracy"),
            "mean_edit_distance": _overall_mean("mean_edit_distance"),
            "mean_reasoning_sim": _overall_mean("mean_reasoning_sim"),
        },
    }

    return summary
