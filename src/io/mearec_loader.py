"""
MEArec recording loader.

Loads a pre-saved MEArec .h5 recording and returns ground-truth templates,
spiketrains, and the SpikeInterface recording extractor.

This complements matlab_loader.py for the simulated data path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_mearec_recording(recording_h5: str | Path) -> Dict:
    """
    Load a MEArec recording from an .h5 file.

    Returns a dict with the same schema used by load_matlab_spikes(), so that
    ClusterManager and pipeline code can be reused without modification:

      {
        "waveforms":          np.ndarray (n_spikes, n_samples)
        "spiketimes":         np.ndarray (n_spikes,)  seconds
        "Fs":                 float
        "overcluster_assigns": np.ndarray (n_spikes,)
        "hierarchy_assigns":  np.ndarray (n_spikes,)
        "hierarchy_tree":     np.ndarray (4, n_merges)
        "gt_assigns":         np.ndarray (n_spikes,)  — None if not loaded
        "n_units":            int
        "recording":          SpikeInterface BaseRecording extractor
        "spiketrains":        list of neo SpikeTrain objects
      }

    Requires: MEArec, spikeinterface, neo
    """
    try:
        import MEArec as mr
        import spikeinterface.full as si
        import spikeinterface.preprocessing as sp
    except ImportError as e:
        raise ImportError("MEArec and spikeinterface are required.") from e

    h5_path = Path(recording_h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"Recording not found: {h5_path}")

    recgen = mr.load_recordings(str(h5_path))
    if hasattr(si, "read_mearec"):
        loaded = si.read_mearec(str(h5_path))
        # spikeinterface>=0.103 returns (recording, sorting) tuple.
        # Older versions may return only the recording extractor.
        if isinstance(loaded, tuple):
            recording = loaded[0]
        else:
            recording = loaded
    else:
        import spikeinterface.extractors as se
        recording = se.MEArecRecordingExtractor(str(h5_path))
    recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = sp.common_reference(recording)

    Fs = recording.get_sampling_frequency()
    spiketrains = recgen.spiketrains
    n_units = len(spiketrains)

    # Build spike array from GT spiketrains
    spike_times_list = []
    gt_assigns_list = []
    for unit_idx, st in enumerate(spiketrains, start=1):
        times = np.array(st.rescale("s").magnitude, dtype=np.float64)
        spike_times_list.append(times)
        gt_assigns_list.append(np.full(len(times), unit_idx, dtype=np.int64))

    spike_times = np.concatenate(spike_times_list)
    gt_assigns = np.concatenate(gt_assigns_list)
    sort_idx = np.argsort(spike_times)
    spike_times = spike_times[sort_idx]
    gt_assigns = gt_assigns[sort_idx]

    # Placeholder waveforms (will be filled by overcluster.py after sorting)
    n_spikes = len(spike_times)
    waveforms = np.zeros((n_spikes, 40), dtype=np.float32)

    # overcluster_assigns and hierarchy_assigns default to gt_assigns until
    # overcluster_recording() overwrites them with sorter output
    overcluster_assigns = gt_assigns.copy()
    hierarchy_assigns = gt_assigns.copy()
    hierarchy_tree = np.empty((4, 0), dtype=np.int64)

    return {
        "waveforms": waveforms,
        "spiketimes": spike_times,
        "Fs": float(Fs),
        "overcluster_assigns": overcluster_assigns,
        "hierarchy_assigns": hierarchy_assigns,
        "hierarchy_tree": hierarchy_tree,
        "gt_assigns": gt_assigns,
        "n_units": n_units,
        "recording": recording,
        "spiketrains": spiketrains,
    }


def load_mearec_from_raw_dir(raw_dir: str | Path) -> Dict:
    """
    Load a channel's data from the numpy arrays saved by overcluster_recording().

    This is the fast path for code that doesn't need the raw recording object —
    it just reads the pre-extracted arrays from raw/.

    Returns the same schema as load_mearec_recording(), minus 'recording' and
    'spiketrains', which are set to None.
    """
    raw_dir = Path(raw_dir)

    def _load(name: str) -> Optional[np.ndarray]:
        p = raw_dir / name
        return np.load(p) if p.exists() else None

    waveforms = _load("waveforms.npy")
    spike_times = _load("spike_times.npy")
    overcluster_assigns = _load("overcluster_assigns.npy")
    hierarchy_assigns = _load("hierarchy_assigns.npy")
    hierarchy_tree = _load("hierarchy_tree.npy")
    gt_assigns = _load("gt_assigns.npy")

    meta: dict = {}
    meta_path = raw_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    Fs = float(meta.get("Fs", 30000.0))

    return {
        "waveforms": waveforms,
        "spiketimes": spike_times,
        "Fs": Fs,
        "overcluster_assigns": overcluster_assigns,
        "hierarchy_assigns": hierarchy_assigns,
        "hierarchy_tree": hierarchy_tree if hierarchy_tree is not None else np.empty((4, 0), dtype=np.int64),
        "gt_assigns": gt_assigns,
        "n_units": int(meta.get("n_units_sorted", 0)),
        "recording": None,
        "spiketrains": None,
        "metadata": meta,
    }
