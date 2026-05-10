"""
Load MATLAB spike sorting data from .mat v7.3 (HDF5) files.

Expected structure:
  spikes/
    waveforms: (45, n_spikes) - per-spike waveforms, spike at sample 18
    spiketimes: (n_spikes, 1) - spike times in SECONDS
    Fs: sampling rate (Hz)
    options/refractory_period: ISI violation threshold (seconds)
    overcluster/assigns: initial overclustering result
    hierarchy/assigns: after hierarchical merging
    hierarchy/tree: merge operations
    curation/assigns: ground truth (human-curated)
    curation/tree: curation merge tree
"""

import numpy as np
import h5py
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import spikeinterface as si
from spikeinterface.core import NumpySorting


def load_matlab_spikes(mat_path: str) -> Dict[str, Any]:
    """
    Load spike data from MATLAB .mat file (supports v7.3 HDF5 via h5py and v5 via scipy.io.loadmat).

    Auto-detects the format and normalizes outputs to a consistent schema.

    Args:
        mat_path: Path to .mat file

    Returns:
        Dictionary with fields:
        - waveforms: (n_spikes, n_samples) array - transposed to row-per-spike
        - spiketimes: (n_spikes,) array in SECONDS
        - Fs: sampling frequency (Hz)
        - refractory_period: ISI violation threshold from options (seconds)
        - overcluster_assigns: (n_spikes,) overclustering labels
        - hierarchy_assigns: (n_spikes,) hierarchical clustering labels
        - hierarchy_tree: (4, n_merges) [src1, src2, similarity, unused]
        - curation_assigns: (n_spikes,) ground truth labels (or None)
        - curation_tree: (4, n_merges) curation tree (or None)
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_path}")
    
    data = {
        'waveforms': None,
        'spiketimes': None,
        'Fs': None,
        'refractory_period': 0.002,  # Default 2ms
        'overcluster_assigns': None,
        'hierarchy_assigns': None,
        'hierarchy_tree': None,
        'curation_assigns': None,
        'curation_tree': None,
    }
    
    # Try HDF5 (v7.3) first; if it fails, fall back to scipy (v5)
    try:
        with h5py.File(mat_path, 'r') as f:
            # Ensure expected group exists
            if 'spikes' not in f:
                raise KeyError("'spikes' group not found in HDF5 file")

            # Read core spike data
            data['Fs'] = float(f['spikes/Fs'][0, 0])
            data['spiketimes'] = f['spikes/spiketimes'][:].flatten()  # (n_spikes,) in SECONDS
            data['waveforms'] = f['spikes/waveforms'][:]  # (45, n_spikes) - will transpose later

            # Read threshold from options
            if 'spikes/options/refractory_period' in f:
                data['refractory_period'] = float(f['spikes/options/refractory_period'][0, 0])

            # Read clustering results
            if 'spikes/overcluster/assigns' in f:
                data['overcluster_assigns'] = f['spikes/overcluster/assigns'][:].flatten()
            if 'spikes/hierarchy/assigns' in f:
                data['hierarchy_assigns'] = f['spikes/hierarchy/assigns'][:].flatten()
            if 'spikes/hierarchy/tree' in f:
                ht = np.array(f['spikes/hierarchy/tree'][:])
                if ht.ndim == 2 and ht.shape[0] != 4 and ht.shape[1] == 4:
                    ht = ht.T
                data['hierarchy_tree'] = ht

            # Read ground truth (may not exist)
            if 'spikes/curation/assigns' in f:
                data['curation_assigns'] = f['spikes/curation/assigns'][:].flatten()
            if 'spikes/curation/tree' in f:
                data['curation_tree'] = f['spikes/curation/tree'][:]
    except Exception:
        # v5 MAT file path via scipy.io.loadmat
        md = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

        # Expect a top-level 'spikes' struct
        if 'spikes' not in md:
            raise KeyError("'spikes' struct not found in MAT file")
        spikes = md['spikes']

        # Access fields robustly (MATLAB struct -> simple object with attributes)
        def get_attr(obj, name, default=None):
            return getattr(obj, name, default)

        # Core fields
        Fs = get_attr(spikes, 'Fs')
        spiketimes = get_attr(spikes, 'spiketimes')
        waveforms = get_attr(spikes, 'waveforms')
        if Fs is None or spiketimes is None or waveforms is None:
            raise KeyError("Missing required fields in 'spikes': Fs, spiketimes, waveforms")

        data['Fs'] = float(np.array(Fs).squeeze())
        data['spiketimes'] = np.array(spiketimes).squeeze().astype(float)
        # MATLAB convention used here is (n_samples, n_spikes)
        wf = np.array(waveforms)
        data['waveforms'] = wf  # will transpose later

        # Options
        options = get_attr(spikes, 'options')
        if options is not None:
            rp = get_attr(options, 'refractory_period')
            if rp is not None:
                data['refractory_period'] = float(np.array(rp).squeeze())

        # Overcluster
        overcluster = get_attr(spikes, 'overcluster')
        if overcluster is not None:
            oc_assigns = get_attr(overcluster, 'assigns')
            if oc_assigns is not None:
                data['overcluster_assigns'] = np.array(oc_assigns).squeeze()
            # Optional extras (ignored if missing)

        # Hierarchy
        hierarchy = get_attr(spikes, 'hierarchy')
        if hierarchy is not None:
            h_assigns = get_attr(hierarchy, 'assigns')
            h_tree = get_attr(hierarchy, 'tree')
            if h_assigns is not None:
                data['hierarchy_assigns'] = np.array(h_assigns).squeeze()
            if h_tree is not None:
                ht = np.array(h_tree)
                if ht.ndim == 2 and ht.shape[0] != 4 and ht.shape[1] == 4:
                    ht = ht.T
                data['hierarchy_tree'] = ht

        # Curation (ground truth)
        curation = get_attr(spikes, 'curation')
        if curation is not None:
            c_assigns = get_attr(curation, 'assigns')
            c_tree = get_attr(curation, 'tree')
            if c_assigns is not None:
                data['curation_assigns'] = np.array(c_assigns).squeeze()
            if c_tree is not None:
                data['curation_tree'] = np.array(c_tree)
    
    # Ensure waveforms shape is (n_spikes, n_samples)
    # Many MAT files store as (n_samples, n_spikes); normalize to rows=spikes
    n_spikes = len(data['spiketimes'])
    wf = np.asarray(data['waveforms'])
    if wf.ndim != 2:
        raise ValueError(f"Waveforms must be 2D, got shape {wf.shape}")
    if wf.shape[0] == n_spikes:
        data['waveforms'] = wf
    elif wf.shape[1] == n_spikes:
        data['waveforms'] = wf.T
    else:
        raise AssertionError(f"Waveforms shape {wf.shape} incompatible with n_spikes={n_spikes}")
    
    # Validation
    n_spikes = len(data['spiketimes'])
    assert data['waveforms'].shape[0] == n_spikes, \
        f"Waveforms shape {data['waveforms'].shape} doesn't match {n_spikes} spikes"
    assert data['overcluster_assigns'].shape[0] == n_spikes, \
        "Overcluster assigns length mismatch"
    assert data['hierarchy_assigns'].shape[0] == n_spikes, \
        "Hierarchy assigns length mismatch"
    
    return data


def build_sorting_from_assigns(spike_times_sec: np.ndarray,
                              assigns: np.ndarray,
                              fs: float) -> si.NumpySorting:
    """
    Build SpikeInterface NumpySorting from spike times and cluster labels.
    
    Args:
        spike_times_sec: Spike times in SECONDS (n_spikes,)
        assigns: Cluster labels (n_spikes,)
        fs: Sampling frequency (Hz)
    
    Returns:
        NumpySorting with spike trains per unit
        
    Note:
        - Converts spike times from seconds to sample indices (multiply by fs)
        - Excludes label 0 (noise/unassigned) if other labels exist
    """
    spike_times_sec = np.asarray(spike_times_sec).flatten()
    assigns = np.asarray(assigns).flatten()
    
    assert len(spike_times_sec) == len(assigns), \
        f"Length mismatch: {len(spike_times_sec)} spikes vs {len(assigns)} labels"
    
    # Convert seconds -> sample indices
    spike_samples = np.round(spike_times_sec * fs).astype(np.int64)
    
    # Get unique labels (exclude 0 if not the only label)
    unique_labels = np.unique(assigns)
    if len(unique_labels) > 1 and 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]
    
    # Build spike trains dict: {unit_id: sample_indices}
    spike_trains = {}
    for label in unique_labels:
        spike_trains[int(label)] = spike_samples[assigns == label]
    
    # Create NumpySorting
    return si.NumpySorting.from_times_labels(
        times_list=[spike_trains[uid] for uid in sorted(spike_trains.keys())],
        labels_list=list(sorted(spike_trains.keys())),
        sampling_frequency=fs
    )


def apply_hierarchy_tree(overcluster_assigns: np.ndarray,
                        hierarchy_tree: np.ndarray) -> np.ndarray:
    """
    Reconstruct hierarchy.assigns by applying tree merges to overcluster.assigns.
    
    Tree format: (4, n_merges) where each column is [src1, src2, similarity, unused]
    Merge rule: src2 is merged INTO src1 (src1 ID is kept)
    
    Args:
        overcluster_assigns: Initial labels (n_spikes,)
        hierarchy_tree: (4, n_merges) merge operations
    
    Returns:
        hierarchy_assigns: Labels after all merges
        
    Note:
        - Merges ordered by similarity: first = highest (0.94), last = lowest (0.01)
        - Fee et al. 1996 hierarchical clustering algorithm
        - 341 overclusters -> 315 merges -> 26 hierarchy clusters
    """
    assigns = overcluster_assigns.copy()
    
    if hierarchy_tree is None or hierarchy_tree.size == 0:
        return assigns
    
    for i in range(hierarchy_tree.shape[1]):
        src1, src2 = int(hierarchy_tree[0, i]), int(hierarchy_tree[1, i])
        # Merge src2 INTO src1 (keep src1 label)
        assigns[assigns == src2] = src1
    
    return assigns


def convert_mat_to_sortings(mat_path: str) -> Tuple[si.NumpySorting, Optional[si.NumpySorting], Dict[str, Any]]:
    """
    Load MATLAB spike data and convert to SpikeInterface sortings.

    Returns:
        initial_sorting: Hierarchy clustering result (pre-curation)
        final_sorting: Ground truth curation result (or None if unavailable)
        meta: Dictionary containing:
            - spiketimes: (n_spikes,) spike times in SECONDS
            - waveforms: (n_spikes, 45) per-spike waveforms
            - Fs: sampling rate (Hz)
            - refractory_period: ISI violation threshold (seconds)
            - overcluster_assigns: (n_spikes,) overclustering labels
            - hierarchy_assigns: (n_spikes,) hierarchy labels
            - hierarchy_tree: (4, n_merges) [src1, src2, similarity, unused]
            - curation_assigns: (n_spikes,) ground truth labels (or None)
            - curation_tree: (4, n_merges) curation tree (or None)
            
    Note:
        Spike times are ALWAYS in seconds in the MATLAB file.
        They are converted to sample indices only for SpikeInterface.
    """
    data = load_matlab_spikes(mat_path)
    
    fs = data['Fs']
    spike_times_sec = data['spiketimes']
    
    # Build initial sorting (hierarchy clustering)
    initial_sorting = build_sorting_from_assigns(
        spike_times_sec, 
        data['hierarchy_assigns'], 
        fs
    )

    # Build final sorting (ground truth) if available
    final_sorting = None
    if data['curation_assigns'] is not None:
        final_sorting = build_sorting_from_assigns(
            spike_times_sec,
            data['curation_assigns'],
            fs
        )

    return initial_sorting, final_sorting, data
