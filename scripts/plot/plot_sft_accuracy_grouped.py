"""
Plot grouped bar charts for unit-test accuracy summaries.

Features:
1) Before/after SFT grouped bars (detailed + per-channel overall)
2) Optional model-level comparison for any number of models:
   - via repeated --model-spec "Label|/path/base.json|/path/ft.json"
   - legacy qwen/gemma flags are still supported

Model-level comparison plot uses one group per model with 3 bars:
- Base-Overall
- FT-CH30
- FT-Overall
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def _to_map(rows: List[Dict], include_all: bool) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for r in rows:
        ch = str(r.get("channel"))
        gt = str(r.get("gt_action"))
        if include_all and gt != "ALL":
            continue
        if not include_all and gt == "ALL":
            continue
        out[(ch, gt)] = float(r.get("accuracy", 0.0))
    return out


def _annotate_bars(ax: plt.Axes, rects, decimals: int = 3) -> None:
    for r in rects:
        h = float(r.get_height())
        ax.text(
            r.get_x() + r.get_width() / 2.0,
            h + 0.015,
            f"{h:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_grouped(
    before: Dict[Tuple[str, str], float],
    after: Dict[Tuple[str, str], float],
    title: str,
    out_path: Path,
) -> None:
    keys = sorted(set(before.keys()) | set(after.keys()))
    if not keys:
        raise ValueError("No rows to plot")

    labels = [f"{ch}-{gt}" for ch, gt in keys]
    bvals = np.array([before.get(k, 0.0) for k in keys], dtype=float)
    avals = np.array([after.get(k, 0.0) for k in keys], dtype=float)

    x = np.arange(len(keys))
    width = 0.38

    fig_w = max(10, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    bars_before = ax.bar(x - width / 2, bvals, width, label="Before SFT", color="#4C78A8")
    bars_after = ax.bar(x + width / 2, avals, width, label="After SFT", color="#F58518")
    _annotate_bars(ax, bars_before)
    _annotate_bars(ax, bars_after)

    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _overall_from_rows(rows: List[Dict]) -> float:
    all_rows = [r for r in rows if str(r.get("gt_action")) == "ALL"]
    if not all_rows:
        raise ValueError("No gt_action=ALL rows found")
    total_n = sum(int(r.get("n", 0) or 0) for r in all_rows)
    total_correct = sum(int(r.get("correct", 0) or 0) for r in all_rows)
    if total_n > 0:
        return float(total_correct) / float(total_n)
    # Fallback if n/correct missing in some legacy files.
    vals = [float(r.get("accuracy", 0.0)) for r in all_rows]
    return float(np.mean(vals))


def _channel_all_accuracy(rows: List[Dict], channel: str) -> float:
    ch = channel.upper()
    for r in rows:
        if str(r.get("gt_action")) == "ALL" and str(r.get("channel", "")).upper() == ch:
            return float(r.get("accuracy", 0.0))
    raise ValueError(f"No channel={channel}, gt_action=ALL row found")


def _safe_load(path_str: Optional[str]) -> Optional[List[Dict]]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return _load_rows(p)


def _plot_model_overall_compare(
    *,
    model_entries: List[Tuple[str, List[Dict], List[Dict]]],
    holdout_channel: str,
    out_path: Path,
) -> None:
    groups: List[str] = []
    series_base_overall: List[float] = []
    series_ft_holdout: List[float] = []
    series_ft_overall: List[float] = []

    for label, base_rows, ft_rows in model_entries:
        groups.append(label)
        series_base_overall.append(_overall_from_rows(base_rows))
        series_ft_holdout.append(_channel_all_accuracy(ft_rows, holdout_channel))
        series_ft_overall.append(_overall_from_rows(ft_rows))

    if not groups:
        return

    x = np.arange(len(groups))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    b0 = ax.bar(x - width, np.array(series_base_overall), width, label="Base-Overall", color="#4C78A8")
    b1 = ax.bar(x, np.array(series_ft_holdout), width, label=f"FT-{holdout_channel.upper()}", color="#54A24B")
    b2 = ax.bar(x + width, np.array(series_ft_overall), width, label="FT-Overall", color="#F58518")
    _annotate_bars(ax, b0)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)

    ax.set_title("Model-Level Overall Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_model_spec(raw: str) -> Tuple[str, str, str]:
    # Format: Label|/path/base.json|/path/ft.json
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 3 or any(not p for p in parts):
        raise ValueError(
            f"Invalid --model-spec: {raw}. Expected format: "
            "Label|/path/base_summary.json|/path/ft_summary.json"
        )
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grouped before/after SFT accuracy bars.")
    parser.add_argument("--before-json", default="output/summary_accuracy.json")
    parser.add_argument("--after-json", default="output/summary_accuracy_ft.json")
    parser.add_argument("--out-detailed", default="output/summary_accuracy_grouped.png")
    parser.add_argument("--out-overall", default="output/summary_accuracy_grouped_overall.png")
    parser.add_argument("--qwen-base-json", default="")
    parser.add_argument("--qwen-ft-json", default="")
    parser.add_argument("--gemma-base-json", default="")
    parser.add_argument("--gemma-ft-json", default="")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Repeatable model entry: Label|/path/base_summary.json|/path/ft_summary.json",
    )
    parser.add_argument("--holdout-channel", default="CH30")
    parser.add_argument("--out-model-overall", default="output/summary_accuracy_model_overall.png")
    args = parser.parse_args()

    before_path = Path(args.before_json)
    after_path = Path(args.after_json)

    before_rows = _load_rows(before_path)
    after_rows = _load_rows(after_path)

    # Detailed by (channel, gt_action)
    b_detail = _to_map(before_rows, include_all=False)
    a_detail = _to_map(after_rows, include_all=False)
    _plot_grouped(
        b_detail,
        a_detail,
        title="Unit Test Accuracy by Channel-Action (Before vs After SFT)",
        out_path=Path(args.out_detailed),
    )

    # Overall by channel (gt_action == ALL)
    b_all = _to_map(before_rows, include_all=True)
    a_all = _to_map(after_rows, include_all=True)
    _plot_grouped(
        b_all,
        a_all,
        title="Unit Test Overall Accuracy by Channel (Before vs After SFT)",
        out_path=Path(args.out_overall),
    )

    print(f"Saved: {args.out_detailed}")
    print(f"Saved: {args.out_overall}")

    # Optional model-level comparison.
    model_entries: List[Tuple[str, List[Dict], List[Dict]]] = []
    for spec in args.model_spec:
        label, base_path, ft_path = _parse_model_spec(spec)
        model_entries.append((label, _load_rows(Path(base_path)), _load_rows(Path(ft_path))))

    # Backward compatibility with legacy fixed args.
    qwen_base_rows = _safe_load(args.qwen_base_json)
    qwen_ft_rows = _safe_load(args.qwen_ft_json)
    gemma_base_rows = _safe_load(args.gemma_base_json)
    gemma_ft_rows = _safe_load(args.gemma_ft_json)
    if qwen_base_rows and qwen_ft_rows:
        model_entries.append(("Qwen3.5", qwen_base_rows, qwen_ft_rows))
    if gemma_base_rows and gemma_ft_rows:
        model_entries.append(("Gemma4", gemma_base_rows, gemma_ft_rows))

    if model_entries:
        _plot_model_overall_compare(
            model_entries=model_entries,
            holdout_channel=args.holdout_channel,
            out_path=Path(args.out_model_overall),
        )
        print(f"Saved: {args.out_model_overall}")


if __name__ == "__main__":
    main()
