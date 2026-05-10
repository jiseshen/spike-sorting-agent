"""Plot RAG vs no-RAG unit-test comparisons for Qwen/Gemma backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_summary(summary_path: Path) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mode_pair(run_dir: Path) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
    no_rag_summary = _load_summary(run_dir / "summary_no_rag.json")
    rag_summary = _load_summary(run_dir / "summary_rag.json")
    no_rag_detail = pd.read_csv(run_dir / "detailed_comparisons_no_rag.csv")
    rag_detail = pd.read_csv(run_dir / "detailed_comparisons_rag.csv")
    return no_rag_summary, rag_summary, no_rag_detail, rag_detail


def _bar_annotate(ax: plt.Axes, rects) -> None:
    for rect in rects:
        h = float(rect.get_height())
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 0.01,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _extract_channel_acc(summary: Dict, channels: List[str]) -> List[float]:
    per_channel = summary.get("per_channel", {})
    vals: List[float] = []
    for ch in channels:
        vals.append(float(per_channel.get(ch, {}).get("overall_accuracy", 0.0)))
    return vals


def plot_channel_bars(
    *,
    model_runs: List[Tuple[str, Dict, Dict]],
    channels: List[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(model_runs), figsize=(6.6 * len(model_runs), 5.0), sharey=True)
    if len(model_runs) == 1:
        axes = [axes]

    x = np.arange(len(channels))
    width = 0.36

    for ax, (label, no_rag_summary, rag_summary) in zip(axes, model_runs):
        no_rag_vals = np.array(_extract_channel_acc(no_rag_summary, channels), dtype=float)
        rag_vals = np.array(_extract_channel_acc(rag_summary, channels), dtype=float)
        b0 = ax.bar(x - width / 2, no_rag_vals, width, label="No-RAG", color="#4C78A8")
        b1 = ax.bar(x + width / 2, rag_vals, width, label="RAG", color="#F58518")
        _bar_annotate(ax, b0)
        _bar_annotate(ax, b1)
        ax.set_title(f"{label}: per-channel accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overall_bars(*, model_runs: List[Tuple[str, Dict, Dict]], out_path: Path) -> None:
    labels = [x[0] for x in model_runs]
    no_rag_vals = np.array(
        [float(x[1].get("overall", {}).get("overall_accuracy", 0.0)) for x in model_runs],
        dtype=float,
    )
    rag_vals = np.array(
        [float(x[2].get("overall", {}).get("overall_accuracy", 0.0)) for x in model_runs],
        dtype=float,
    )

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    b0 = ax.bar(x - width / 2, no_rag_vals, width, label="No-RAG", color="#4C78A8")
    b1 = ax.bar(x + width / 2, rag_vals, width, label="RAG", color="#F58518")
    _bar_annotate(ax, b0)
    _bar_annotate(ax, b1)
    ax.set_title("Overall accuracy across all channels")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _to_flat_sequence(df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    channel_order = {ch: i for i, ch in enumerate(channels)}
    out = df.copy()
    out["channel"] = out["channel"].astype(str).str.upper()
    out["step"] = pd.to_numeric(out["step"], errors="coerce").fillna(-1).astype(int)
    out["match"] = out["match"].astype(bool).astype(float)
    out["channel_order"] = out["channel"].map(channel_order).fillna(10**6).astype(int)
    out = out.sort_values(["channel_order", "step"], kind="mergesort").reset_index(drop=True)
    out["global_step"] = np.arange(1, len(out) + 1)
    return out


def _channel_boundaries(flat_df: pd.DataFrame, channels: List[str]) -> Tuple[List[int], List[Tuple[float, str]]]:
    boundaries: List[int] = []
    labels: List[Tuple[float, str]] = []
    start = 1
    for ch in channels:
        ch_df = flat_df[flat_df["channel"] == ch]
        n = len(ch_df)
        if n == 0:
            continue
        end = start + n - 1
        labels.append(((start + end) / 2.0, ch))
        boundaries.append(end)
        start = end + 1
    return boundaries[:-1], labels


def plot_learning_curve(
    *,
    model_label: str,
    no_rag_detail: pd.DataFrame,
    rag_detail: pd.DataFrame,
    channels: List[str],
    rolling_window: int,
    out_path: Path,
) -> None:
    no_df = _to_flat_sequence(no_rag_detail, channels)
    rag_df = _to_flat_sequence(rag_detail, channels)

    no_curve = no_df["match"].rolling(window=rolling_window, min_periods=1, center=True).mean()
    rag_curve = rag_df["match"].rolling(window=rolling_window, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.plot(no_df["global_step"], no_curve, label="No-RAG", color="#4C78A8", linewidth=2.0)
    ax.plot(rag_df["global_step"], rag_curve, label="RAG", color="#F58518", linewidth=2.0)
    ax.set_title(f"{model_label}: learning curve (rolling mean, window={rolling_window})")
    ax.set_xlabel("Flattened step index (CH3 -> CH20 -> CH30 -> CH31)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()

    boundaries, labels = _channel_boundaries(no_df, channels)
    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    y_text = 0.02
    for xmid, ch in labels:
        ax.text(xmid, y_text, ch, ha="center", va="bottom", fontsize=9, alpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_overall_table(model_runs: List[Tuple[str, Dict, Dict]], out_path: Path) -> None:
    rows = []
    for label, no_rag_summary, rag_summary in model_runs:
        rows.append(
            {
                "model": label,
                "mode": "no_rag",
                "overall_accuracy": float(no_rag_summary.get("overall", {}).get("overall_accuracy", 0.0)),
                "n_total": int(no_rag_summary.get("overall", {}).get("n_total", 0)),
            }
        )
        rows.append(
            {
                "model": label,
                "mode": "rag",
                "overall_accuracy": float(rag_summary.get("overall", {}).get("overall_accuracy", 0.0)),
                "n_total": int(rag_summary.get("overall", {}).get("n_total", 0)),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen/Gemma RAG-vs-no-RAG unit-test comparisons.")
    parser.add_argument("--qwen-dir", default="output/rag_backbone_eval/qwen35_4b")
    parser.add_argument("--gemma-dir", default="output/rag_backbone_eval/gemma4_e4b")
    parser.add_argument("--channels", nargs="+", default=["CH3", "CH20", "CH30", "CH31"])
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--out-dir", default="output/rag_backbone_eval/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channels = [str(ch).upper() for ch in args.channels]
    out_dir = Path(args.out_dir)

    q_no, q_rag, q_no_df, q_rag_df = _load_mode_pair(Path(args.qwen_dir))
    g_no, g_rag, g_no_df, g_rag_df = _load_mode_pair(Path(args.gemma_dir))

    model_runs = [
        ("Qwen3.5-4B", q_no, q_rag),
        ("Gemma4-E4B", g_no, g_rag),
    ]

    plot_channel_bars(
        model_runs=model_runs,
        channels=channels,
        out_path=out_dir / "per_channel_rag_vs_no_rag.png",
    )
    plot_overall_bars(
        model_runs=model_runs,
        out_path=out_dir / "overall_rag_vs_no_rag_qwen_gemma.png",
    )
    plot_learning_curve(
        model_label="Qwen3.5-4B",
        no_rag_detail=q_no_df,
        rag_detail=q_rag_df,
        channels=channels,
        rolling_window=args.rolling_window,
        out_path=out_dir / "learning_curve_qwen35_4b.png",
    )
    plot_learning_curve(
        model_label="Gemma4-E4B",
        no_rag_detail=g_no_df,
        rag_detail=g_rag_df,
        channels=channels,
        rolling_window=args.rolling_window,
        out_path=out_dir / "learning_curve_gemma4_e4b.png",
    )
    _save_overall_table(model_runs, out_dir / "overall_accuracy_table.csv")

    print(f"Saved: {out_dir / 'per_channel_rag_vs_no_rag.png'}")
    print(f"Saved: {out_dir / 'overall_rag_vs_no_rag_qwen_gemma.png'}")
    print(f"Saved: {out_dir / 'learning_curve_qwen35_4b.png'}")
    print(f"Saved: {out_dir / 'learning_curve_gemma4_e4b.png'}")
    print(f"Saved: {out_dir / 'overall_accuracy_table.csv'}")


if __name__ == "__main__":
    main()
