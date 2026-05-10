"""
Visualize simulated MEArec ground-truth clusters (pre-curation style).

For each channel under:
  output/<setting_id>/<channel_id>/raw/recording.h5

this script:
  1) loads GT spiketrains from MEArec
  2) loads filtered recording through SpikeInterface
  3) extracts per-GT-unit snippets on one display channel
  4) renders baseline-style cluster grids

Output:
  output/visualizations/simulated_gt/<setting_id>/<channel_id>/ground_truth.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_cluster_waveforms(
    ax,
    waveforms: np.ndarray,
    cluster_id: int,
    total_spikes: int,
    sampling_rate: float,
    max_waveforms: int,
    rng: np.random.Generator,
) -> None:
    n_snippets, n_samples = waveforms.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000.0

    if n_snippets > max_waveforms:
        idx = rng.choice(n_snippets, max_waveforms, replace=False)
        plot_wf = waveforms[idx]
    else:
        plot_wf = waveforms

    for wf in plot_wf:
        ax.plot(time_ms, wf, color="seagreen", alpha=0.2, linewidth=0.5)

    if n_snippets > 0:
        median_wf = np.median(waveforms, axis=0)
        ax.plot(time_ms, median_wf, color="darkgreen", linewidth=2)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Amplitude (uV)", fontsize=8)
    ax.set_title(
        f"GT Unit {cluster_id}\n(snippets={n_snippets}, spikes={total_spikes})",
        fontsize=9,
        fontweight="bold",
    )
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def _discover_channels(setting_dir: Path) -> list[str]:
    return sorted(
        d.name
        for d in setting_dir.iterdir()
        if d.is_dir() and d.name.startswith("ch_") and (d / "raw" / "recording.h5").exists()
    )


def _load_recording_and_gt(rec_path: Path):
    import MEArec as mr
    import spikeinterface.full as si
    import spikeinterface.preprocessing as sp

    recgen = mr.load_recordings(str(rec_path))
    if hasattr(si, "read_mearec"):
        loaded = si.read_mearec(str(rec_path))
        recording = loaded[0] if isinstance(loaded, tuple) else loaded
    else:
        import spikeinterface.extractors as se

        recording = se.MEArecRecordingExtractor(str(rec_path))

    recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = sp.common_reference(recording)
    return recording, recgen.spiketrains


def _extract_gt_waveforms(
    recording,
    gt_spiketrains,
    snippet_t1: int,
    snippet_t2: int,
    display_channel_index: int,
    max_spikes_per_unit: int,
    rng: np.random.Generator,
):
    fs = float(recording.get_sampling_frequency())
    channel_ids = list(recording.get_channel_ids())
    if display_channel_index < 0 or display_channel_index >= len(channel_ids):
        raise ValueError(
            f"display_channel_index={display_channel_index} out of range "
            f"(n_channels={len(channel_ids)})"
        )
    ch_id = channel_ids[display_channel_index]
    n_samples = snippet_t1 + snippet_t2
    n_frames = int(recording.get_num_frames())

    per_unit_waveforms: dict[int, np.ndarray] = {}
    per_unit_counts: dict[int, int] = {}
    for unit_id, st in enumerate(gt_spiketrains, start=1):
        spike_frames = np.asarray((st.rescale("s").magnitude * fs).round(), dtype=np.int64)
        valid = spike_frames[(spike_frames >= snippet_t1) & (spike_frames + snippet_t2 < n_frames)]
        per_unit_counts[unit_id] = int(valid.size)
        if valid.size == 0:
            continue

        if valid.size > max_spikes_per_unit:
            chosen = rng.choice(valid, size=max_spikes_per_unit, replace=False)
            chosen.sort()
        else:
            chosen = valid

        wfs = np.zeros((len(chosen), n_samples), dtype=np.float32)
        for i, frame in enumerate(chosen):
            seg = recording.get_traces(
                start_frame=int(frame - snippet_t1),
                end_frame=int(frame + snippet_t2),
                channel_ids=[ch_id],
            )
            wfs[i] = seg[:, 0].astype(np.float32, copy=False)

        per_unit_waveforms[unit_id] = wfs

    return per_unit_waveforms, per_unit_counts, fs


def _plot_channel_gt(
    setting_id: str,
    channel_id: str,
    per_unit_waveforms: dict[int, np.ndarray],
    per_unit_counts: dict[int, int],
    sampling_rate: float,
    output_path: Path,
    max_waveforms_plot: int,
    seed: int,
) -> bool:
    cluster_ids = sorted(uid for uid, wfs in per_unit_waveforms.items() if wfs.shape[0] > 0)
    if not cluster_ids:
        print(f"  [skip] {channel_id}: no valid GT snippets")
        return False

    n_clusters = len(cluster_ids)
    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows))
    fig.suptitle(
        f"{setting_id}/{channel_id} - Ground Truth Clusters - {n_clusters} units",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    axes = np.array([axes]) if n_clusters == 1 else np.asarray(axes).flatten()
    rng = np.random.default_rng(seed)

    for i, uid in enumerate(cluster_ids):
        _plot_cluster_waveforms(
            ax=axes[i],
            waveforms=per_unit_waveforms[uid],
            cluster_id=uid,
            total_spikes=per_unit_counts.get(uid, 0),
            sampling_rate=sampling_rate,
            max_waveforms=max_waveforms_plot,
            rng=rng,
        )

    for i in range(n_clusters, len(axes)):
        axes[i].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MEArec ground-truth clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--setting-id", required=True, help="Setting ID (e.g. setting_003).")
    parser.add_argument("--output-dir", default="output", help="Root output directory.")
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional channel IDs (e.g. ch_000 ch_001). Default: auto-discover.",
    )
    parser.add_argument(
        "--viz-root",
        default="output/visualizations/simulated_gt",
        help="Root visualization output dir.",
    )
    parser.add_argument("--snippet-t1", type=int, default=20, help="Samples before spike.")
    parser.add_argument("--snippet-t2", type=int, default=20, help="Samples after spike.")
    parser.add_argument(
        "--display-channel-index",
        type=int,
        default=0,
        help="Which recording channel to display in waveforms.",
    )
    parser.add_argument(
        "--max-spikes-per-unit",
        type=int,
        default=1200,
        help="Max GT spikes sampled per unit for snippet extraction.",
    )
    parser.add_argument(
        "--max-waveforms-plot",
        type=int,
        default=500,
        help="Max waveforms drawn per unit.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    setting_dir = Path(args.output_dir) / args.setting_id
    if not setting_dir.exists():
        raise FileNotFoundError(f"Setting directory not found: {setting_dir}")

    channels = args.channels if args.channels else _discover_channels(setting_dir)
    if not channels:
        raise RuntimeError(f"No channels with recording.h5 under: {setting_dir}")

    print("=" * 80)
    print(f"Simulated Ground-Truth Visualization: {args.setting_id}")
    print(f"Channels: {len(channels)}")
    print("=" * 80)

    rng = np.random.default_rng(args.seed)
    for ch in channels:
        rec_path = setting_dir / ch / "raw" / "recording.h5"
        if not rec_path.exists():
            print(f"  [skip] {ch}: recording.h5 missing")
            continue

        recording, gt_spiketrains = _load_recording_and_gt(rec_path)
        per_unit_waveforms, per_unit_counts, fs = _extract_gt_waveforms(
            recording=recording,
            gt_spiketrains=gt_spiketrains,
            snippet_t1=args.snippet_t1,
            snippet_t2=args.snippet_t2,
            display_channel_index=args.display_channel_index,
            max_spikes_per_unit=args.max_spikes_per_unit,
            rng=rng,
        )

        out_path = Path(args.viz_root) / args.setting_id / ch / "ground_truth.png"
        _plot_channel_gt(
            setting_id=args.setting_id,
            channel_id=ch,
            per_unit_waveforms=per_unit_waveforms,
            per_unit_counts=per_unit_counts,
            sampling_rate=fs,
            output_path=out_path,
            max_waveforms_plot=args.max_waveforms_plot,
            seed=args.seed,
        )

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()

