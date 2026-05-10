"""
Spike sorting + overclustering for one simulated channel.

Reads recording.h5 produced by generator.py, runs the configured sorter via
SpikeInterface, then extracts:
  - waveforms.npy            (n_spikes, n_samples)
  - spike_times.npy          (n_spikes,)  seconds
  - overcluster_assigns.npy  (n_spikes,)  fine-grained cluster labels
  - hierarchy_assigns.npy    (n_spikes,)  coarser merged labels
  - hierarchy_tree.npy       (4, n_merges) merge tree [dst, src, similarity, score]
  - gt_assigns.npy           (n_spikes,)  ground-truth unit labels from MEArec

These arrays match the schema expected by ClusterManager.__init__().
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .setting import SettingConfig


def overcluster_recording(
    cfg: SettingConfig,
    channel_idx: int,
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    """
    Run spike sorting + overclustering on a pre-generated MEArec recording.

    Args:
        cfg: SettingConfig for this setting.
        channel_idx: Channel index.
        output_dir: Root output directory.
        force: Re-run even if outputs already exist.

    Returns:
        Path to the raw/ directory containing the numpy arrays.
    """
    try:
        import MEArec as mr
        import spikeinterface.full as si
        import spikeinterface.sorters as ss
        import spikeinterface.preprocessing as sp
    except ImportError as e:
        raise ImportError(
            "spikeinterface and MEArec are required. "
            "Install with: pip install spikeinterface MEArec"
        ) from e

    channel_id = f"ch_{channel_idx:03d}"
    raw_dir = Path(output_dir) / cfg.setting_id / channel_id / "raw"
    rec_path = raw_dir / "recording.h5"

    if not rec_path.exists():
        raise FileNotFoundError(
            f"recording.h5 not found at {rec_path}. "
            "Run 01_simulate.py first."
        )

    arrays_complete = all(
        (raw_dir / f).exists()
        for f in ("waveforms.npy", "spike_times.npy", "overcluster_assigns.npy",
                  "gt_assigns.npy", "hierarchy_assigns.npy", "hierarchy_tree.npy")
    )
    if arrays_complete and not force:
        print(f"  [skip] Overclustering already done for {channel_id}")
        return raw_dir

    # --- Load MEArec recording via SpikeInterface ---
    recgen = mr.load_recordings(str(rec_path))
    if hasattr(si, "read_mearec"):
        loaded = si.read_mearec(str(rec_path))
        # spikeinterface>=0.103 returns (recording, sorting) tuple.
        # Older versions may return only the recording extractor.
        if isinstance(loaded, tuple):
            recording = loaded[0]
        else:
            recording = loaded
    else:
        import spikeinterface.extractors as se
        recording = se.MEArecRecordingExtractor(str(rec_path))
    recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = sp.common_reference(recording)

    # --- Ground-truth spike trains from MEArec ---
    gt_spiketrains = recgen.spiketrains   # list of neo SpikeTrain objects
    Fs = recording.get_sampling_frequency()

    # --- Run sorter (overclustering) ---
    sorter_params: dict = {
        "detect_threshold": cfg.detect_threshold,
        "snippet_T1": cfg.snippet_T1,
        "snippet_T2": cfg.snippet_T2,
        "filter": False,            # already filtered above
    }
    sorting = ss.run_sorter(
        cfg.sorter_name,
        recording,
        str(raw_dir / "sorting_output"),
        remove_existing_folder=True,
        **sorter_params,
    )

    unit_ids = sorting.get_unit_ids()
    n_samples = cfg.snippet_T1 + cfg.snippet_T2

    # Guard: sorter may return no units for difficult/noisy channels.
    if len(unit_ids) == 0:
        spike_times = np.empty((0,), dtype=np.float64)
        overcluster_assigns = np.empty((0,), dtype=np.int64)
        hierarchy_assigns = np.empty((0,), dtype=np.int64)
        hierarchy_tree = np.empty((4, 0), dtype=np.float64)
        waveforms = np.zeros((0, n_samples), dtype=np.float32)
        gt_assigns = np.empty((0,), dtype=np.int64)

        np.save(raw_dir / "waveforms.npy", waveforms)
        np.save(raw_dir / "spike_times.npy", spike_times)
        np.save(raw_dir / "overcluster_assigns.npy", overcluster_assigns)
        np.save(raw_dir / "hierarchy_assigns.npy", hierarchy_assigns)
        np.save(raw_dir / "hierarchy_tree.npy", hierarchy_tree)
        np.save(raw_dir / "gt_assigns.npy", gt_assigns)

        with open(raw_dir / "metadata.json") as f:
            meta = json.load(f)
        meta["Fs"] = float(Fs)
        meta["n_units_sorted"] = 0
        meta["n_spikes"] = 0
        with open(raw_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [done] Overclustered {channel_id}: 0 units, 0 spikes")
        return raw_dir

    # Prepare per-unit spike trains early so waveform extraction does not
    # subsample units (default cap=500) and break downstream shape alignment.
    spike_times_list = []
    overcluster_assigns_list = []
    max_unit_spikes = 0
    for uid in unit_ids:
        frames = sorting.get_unit_spike_train(uid)
        max_unit_spikes = max(max_unit_spikes, int(len(frames)))
        spike_times_list.append(frames / Fs)
        overcluster_assigns_list.append(np.full(len(frames), uid, dtype=np.int64))

    # --- Extract waveforms ---
    wf_cache_dir = raw_dir / "waveforms_cache"
    # spikeinterface>=0.101 compatibility layer does not support
    # overwrite/load_if_exists flags in extract_waveforms().
    # We control cache lifecycle here to keep behavior deterministic.
    if wf_cache_dir.exists():
        shutil.rmtree(wf_cache_dir)

    we = si.extract_waveforms(
        recording,
        sorting,
        folder=str(wf_cache_dir),
        ms_before=cfg.snippet_T1 / Fs * 1000,
        ms_after=cfg.snippet_T2 / Fs * 1000,
        max_spikes_per_unit=max_unit_spikes,
    )

    if spike_times_list:
        spike_times = np.concatenate(spike_times_list)
        overcluster_assigns = np.concatenate(overcluster_assigns_list)
    else:
        spike_times = np.empty((0,), dtype=np.float64)
        overcluster_assigns = np.empty((0,), dtype=np.int64)

    sort_idx = np.argsort(spike_times)
    spike_times = spike_times[sort_idx]
    overcluster_assigns = overcluster_assigns[sort_idx]

    n_spikes = len(spike_times)
    waveforms = np.zeros((n_spikes, n_samples), dtype=np.float32)
    unit_mean_waveforms: dict[int, np.ndarray] = {}
    unit_spike_counts: dict[int, int] = {}
    for uid in unit_ids:
        uid_mask = overcluster_assigns == uid
        uid_wv = we.get_waveforms(uid)      # (n, n_samples, n_ch)
        if uid_wv.shape[0] == 0:
            continue
        uid_wv_ch0 = uid_wv[:, :, 0]
        n_target = int(np.count_nonzero(uid_mask))
        n_source = int(uid_wv_ch0.shape[0])
        if n_source != n_target:
            n_copy = min(n_source, n_target)
            print(
                f"  [warn] Unit {uid}: waveform count {n_source} != spike count {n_target}; "
                f"copying first {n_copy} waveforms."
            )
            uid_positions = np.flatnonzero(uid_mask)
            if n_copy > 0:
                waveforms[uid_positions[:n_copy]] = uid_wv_ch0[:n_copy]
        else:
            waveforms[uid_mask] = uid_wv_ch0
        unit_mean_waveforms[int(uid)] = uid_wv_ch0.mean(axis=0).astype(np.float32)
        unit_spike_counts[int(uid)] = int(uid_wv_ch0.shape[0])

    # --- Build gt_assigns by matching spikes to MEArec GT ---
    gt_assigns = _build_gt_assigns(spike_times, gt_spiketrains, Fs, n_spikes)

    # --- hierarchy_assigns and hierarchy_tree ---
    hierarchy_tree = _build_hierarchy_tree_from_unit_waveforms(
        unit_mean_waveforms=unit_mean_waveforms,
        unit_spike_counts=unit_spike_counts,
        enabled=cfg.hierarchy_enabled,
        min_similarity=cfg.hierarchy_min_similarity,
        max_merges=cfg.hierarchy_max_merges,
        similarity_metric=cfg.hierarchy_similarity_metric,
    )
    hierarchy_assigns = _apply_hierarchy_tree(overcluster_assigns, hierarchy_tree)

    # --- Save arrays ---
    np.save(raw_dir / "waveforms.npy", waveforms)
    np.save(raw_dir / "spike_times.npy", spike_times)
    np.save(raw_dir / "overcluster_assigns.npy", overcluster_assigns)
    np.save(raw_dir / "hierarchy_assigns.npy", hierarchy_assigns)
    np.save(raw_dir / "hierarchy_tree.npy", hierarchy_tree)
    np.save(raw_dir / "gt_assigns.npy", gt_assigns)

    with open(raw_dir / "metadata.json") as f:
        meta = json.load(f)
    meta["Fs"] = float(Fs)
    meta["n_units_sorted"] = int(len(unit_ids))
    meta["n_spikes"] = int(n_spikes)
    meta["n_hierarchy_merges"] = int(hierarchy_tree.shape[1])
    meta["n_hierarchy_clusters"] = int(len(np.unique(hierarchy_assigns[hierarchy_assigns != 0])))
    with open(raw_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [done] Overclustered {channel_id}: {len(unit_ids)} units, {n_spikes} spikes")
    return raw_dir


def _build_gt_assigns(
    spike_times: np.ndarray,
    gt_spiketrains,
    Fs: float,
    n_spikes: int,
    tolerance_s: float = 0.0005,   # 0.5 ms matching window
) -> np.ndarray:
    """Match detected spikes to GT spiketrains within ±tolerance_s."""
    gt_assigns = np.zeros(n_spikes, dtype=np.int64)

    for gt_unit_idx, st in enumerate(gt_spiketrains, start=1):
        gt_times = np.array(st.rescale("s").magnitude, dtype=np.float64)
        for i, t in enumerate(spike_times):
            diffs = np.abs(gt_times - t)
            if diffs.min() <= tolerance_s:
                gt_assigns[i] = gt_unit_idx   # last GT unit wins on overlap

    return gt_assigns


def _build_hierarchy_tree_from_unit_waveforms(
    unit_mean_waveforms: dict[int, np.ndarray],
    unit_spike_counts: dict[int, int],
    enabled: bool,
    min_similarity: float,
    max_merges: int | None,
    similarity_metric: str = "pearson",
) -> np.ndarray:
    """
    Build a merge history tree by greedy agglomeration on unit mean waveforms.

    Each merge record is [dst_cluster_id, src_cluster_id, similarity, score].
    """
    if (not enabled) or len(unit_mean_waveforms) <= 1:
        return np.empty((4, 0), dtype=np.float64)

    centroids: dict[int, np.ndarray] = {
        int(uid): np.asarray(wf, dtype=np.float64)
        for uid, wf in unit_mean_waveforms.items()
    }
    counts: dict[int, int] = {
        int(uid): int(unit_spike_counts.get(uid, 0))
        for uid in centroids
    }
    active: set[int] = set(centroids.keys())
    merges: list[list[float]] = []

    while len(active) >= 2:
        active_list = sorted(active)
        best_pair: tuple[int, int] | None = None
        best_sim = -np.inf

        for i, cid_a in enumerate(active_list):
            wa = centroids[cid_a]
            for cid_b in active_list[i + 1:]:
                wb = centroids[cid_b]
                sim = _waveform_similarity(wa, wb, similarity_metric)
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (cid_a, cid_b)

        if best_pair is None:
            break
        if best_sim < float(min_similarity):
            break
        if max_merges is not None and len(merges) >= int(max_merges):
            break

        cid_a, cid_b = best_pair
        count_a = counts[cid_a]
        count_b = counts[cid_b]

        if count_a > count_b:
            dst, src = cid_a, cid_b
        elif count_b > count_a:
            dst, src = cid_b, cid_a
        else:
            dst, src = (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)

        merges.append([float(dst), float(src), float(best_sim), 0.0])

        total = max(counts[dst] + counts[src], 1)
        centroids[dst] = (centroids[dst] * counts[dst] + centroids[src] * counts[src]) / total
        counts[dst] = total

        del centroids[src]
        del counts[src]
        active.remove(src)

    if not merges:
        return np.empty((4, 0), dtype=np.float64)

    return np.asarray(merges, dtype=np.float64).T


def _waveform_similarity(a: np.ndarray, b: np.ndarray, metric: str = "pearson") -> float:
    """Similarity between two 1D waveforms."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    metric = metric.lower()
    if metric == "pearson":
        a = a - a.mean()
        b = b - b.mean()
    elif metric == "cosine":
        pass
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return -1.0

    sim = float(np.dot(a, b) / (na * nb))
    if np.isnan(sim):
        return -1.0
    return float(np.clip(sim, -1.0, 1.0))


def _apply_hierarchy_tree(overcluster_assigns: np.ndarray, hierarchy_tree: np.ndarray) -> np.ndarray:
    """Apply merge tree records to overcluster labels to obtain hierarchy labels."""
    assigns = overcluster_assigns.copy()
    if hierarchy_tree.size == 0:
        return assigns

    for i in range(hierarchy_tree.shape[1]):
        dst = int(hierarchy_tree[0, i])
        src = int(hierarchy_tree[1, i])
        assigns[assigns == src] = dst
    return assigns
