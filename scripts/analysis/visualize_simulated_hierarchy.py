"""
Visualize simulated pre-curation hierarchy clusters.

For each channel under:
  output/<setting_id>/<channel_id>/raw/

loads:
  - waveforms.npy
  - spike_times.npy
  - hierarchy_assigns.npy
  - metadata.json (for Fs)

and renders a baseline-style cluster grid image:
  output/visualizations/simulated/<setting_id>/<channel_id>/baseline.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_cluster_waveforms(
    ax,
    waveforms: np.ndarray,
    cluster_id: int,
    sampling_rate: float,
    max_waveforms: int,
    rng: np.random.Generator,
) -> None:
    n_spikes, n_samples = waveforms.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000.0

    if n_spikes > max_waveforms:
        idx = rng.choice(n_spikes, max_waveforms, replace=False)
        plot_wf = waveforms[idx]
    else:
        plot_wf = waveforms

    for wf in plot_wf:
        ax.plot(time_ms, wf, color="steelblue", alpha=0.2, linewidth=0.5)

    if n_spikes > 0:
        median_wf = np.median(waveforms, axis=0)
        ax.plot(time_ms, median_wf, color="darkblue", linewidth=2)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude (uV)", fontsize=8)
    ax.set_title(f"Cluster {cluster_id}\n({n_spikes} spikes)", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def _create_grid_figure(
    channel_id: str,
    setting_id: str,
    waveforms_all: np.ndarray,
    hierarchy_assigns: np.ndarray,
    sampling_rate: float,
    output_path: Path,
    max_waveforms: int,
    seed: int,
) -> None:
    cluster_ids = sorted(int(cid) for cid in np.unique(hierarchy_assigns) if cid > 0)
    if not cluster_ids:
        print(f"  [skip] {channel_id}: no positive hierarchy clusters")
        return

    n_clusters = len(cluster_ids)
    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))
    fig.suptitle(
        f"{setting_id}/{channel_id} - Baseline (Hierarchy Clustering) - {n_clusters} clusters",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if n_clusters == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes).flatten()

    rng = np.random.default_rng(seed)
    for i, cid in enumerate(cluster_ids):
        mask = hierarchy_assigns == cid
        wf = waveforms_all[mask]
        _plot_cluster_waveforms(
            ax=axes[i],
            waveforms=wf,
            cluster_id=cid,
            sampling_rate=sampling_rate,
            max_waveforms=max_waveforms,
            rng=rng,
        )

    for i in range(n_clusters, len(axes)):
        axes[i].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {output_path}")


def _discover_channels(setting_dir: Path) -> list[str]:
    return sorted(
        d.name
        for d in setting_dir.iterdir()
        if d.is_dir() and d.name.startswith("ch_") and (d / "raw" / "hierarchy_assigns.npy").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize simulated baseline hierarchy clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--setting-id", required=True, help="Setting ID (e.g. setting_003).")
    parser.add_argument("--output-dir", default="output", help="Root output directory.")
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional list of channel IDs (e.g. ch_000 ch_001). Default: all available channels.",
    )
    parser.add_argument(
        "--viz-root",
        default="output/visualizations/simulated",
        help="Root directory for visualization outputs.",
    )
    parser.add_argument("--max-waveforms", type=int, default=500, help="Max waveforms per cluster to draw.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for waveform subsampling.")
    args = parser.parse_args()

    setting_dir = Path(args.output_dir) / args.setting_id
    if not setting_dir.exists():
        raise FileNotFoundError(f"Setting directory not found: {setting_dir}")

    channels = args.channels if args.channels else _discover_channels(setting_dir)
    if not channels:
        raise RuntimeError(f"No channels with hierarchy_assigns.npy found under: {setting_dir}")

    print("=" * 80)
    print(f"Simulated Hierarchy Visualization: {args.setting_id}")
    print(f"Channels: {len(channels)}")
    print("=" * 80)

    for ch in channels:
        raw_dir = setting_dir / ch / "raw"
        if not raw_dir.exists():
            print(f"  [skip] {ch}: raw dir missing")
            continue

        wf_path = raw_dir / "waveforms.npy"
        hs_path = raw_dir / "hierarchy_assigns.npy"
        if not wf_path.exists() or not hs_path.exists():
            print(f"  [skip] {ch}: missing waveforms/hierarchy arrays")
            continue

        waveforms = np.load(wf_path)
        hierarchy_assigns = np.load(hs_path)

        meta_path = raw_dir / "metadata.json"
        sampling_rate = 30000.0
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            sampling_rate = float(meta.get("Fs", sampling_rate))

        out_path = Path(args.viz_root) / args.setting_id / ch / "baseline.png"
        _create_grid_figure(
            channel_id=ch,
            setting_id=args.setting_id,
            waveforms_all=waveforms,
            hierarchy_assigns=hierarchy_assigns,
            sampling_rate=sampling_rate,
            output_path=out_path,
            max_waveforms=args.max_waveforms,
            seed=args.seed,
        )

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()

