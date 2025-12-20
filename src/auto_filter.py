"""
Automated pre-filtering utilities for spike sorting curation.

These functions implement rule-based filtering that doesn't require LLM analysis.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FilterAction:
    """Record of an automated filtering action."""
    cluster_id: int
    action: str  # 'm 0 X' for discard
    reasoning: str
    n_spikes: int


def automatic_size_filter(assigns: np.ndarray, 
                         threshold: int = 500) -> Tuple[List[int], List[FilterAction]]:
    """
    Automatically discard all clusters with < threshold spikes.
    
    This is Phase 0 filtering based on the principle that after hierarchical
    similarity-based clustering, small clusters are remnants/noise.
    
    Args:
        assigns: (n_spikes,) cluster assignments
        threshold: Minimum spike count (default: 500)
    
    Returns:
        kept_clusters: List of cluster IDs that passed filtering
        actions: List of FilterAction records for discarded clusters
    """
    unique_clusters = np.unique(assigns)
    # Exclude cluster 0 (already noise)
    unique_clusters = unique_clusters[unique_clusters != 0]
    
    kept_clusters = []
    actions = []
    
    for cluster_id in unique_clusters:
        n_spikes = (assigns == cluster_id).sum()
        
        if n_spikes < threshold:
            # Discard
            action = FilterAction(
                cluster_id=int(cluster_id),
                action=f"m 0 {cluster_id}",
                reasoning="Event count is too low for individual clusters at end of hierarchical merging.",
                n_spikes=int(n_spikes)
            )
            actions.append(action)
        else:
            # Keep for further analysis
            kept_clusters.append(int(cluster_id))
    
    return kept_clusters, actions


def assess_waveform_shape(waveforms: np.ndarray, 
                         fs: float = 30000) -> Dict[str, any]:
    """
    Assess waveform shape for physiological validity.
    
    Rule-based checks that can filter out clearly invalid waveforms
    before sending to LLM.
    
    Args:
        waveforms: (n_spikes, n_samples) waveform array
        fs: Sampling frequency (Hz)
    
    Returns:
        Dict with:
            - is_valid: bool (passes all checks)
            - violations: List[str] (reasons if invalid)
            - n_phases: int
            - peak_to_trough_ms: float
            - baseline_offset: float
    """
    mean_wf = waveforms.mean(axis=0)
    
    # Find peak and trough
    peak_idx = np.argmax(mean_wf)
    trough_idx = np.argmin(mean_wf)
    
    # Peak-to-trough time
    pt_samples = abs(trough_idx - peak_idx)
    pt_time_ms = pt_samples / fs * 1000
    
    # Count zero-crossings (phases)
    # A phase is a region between zero crossings
    zero_crossings = np.where(np.diff(np.sign(mean_wf)))[0]
    n_phases = len(zero_crossings) + 1
    
    # Baseline offset (should start near zero)
    baseline_offset = abs(mean_wf[0]) / abs(mean_wf.min())  # Normalized
    
    # Check descent speed (should be sharp, not gradual)
    # Measure how many samples it takes to go from 10% to 90% of min
    min_val = mean_wf.min()
    threshold_10 = 0.1 * min_val
    threshold_90 = 0.9 * min_val
    
    # Find first crossing of thresholds
    cross_10 = np.where(mean_wf < threshold_10)[0]
    cross_90 = np.where(mean_wf < threshold_90)[0]
    
    if len(cross_10) > 0 and len(cross_90) > 0:
        descent_samples = cross_90[0] - cross_10[0]
        descent_time_ms = descent_samples / fs * 1000
    else:
        descent_time_ms = np.nan
    
    # Apply rules
    violations = []
    
    if n_phases > 3:
        violations.append(f"Too many phases ({n_phases} > 3)")
    
    if n_phases < 2:
        violations.append(f"Monophasic waveform ({n_phases} phase)")
    
    if pt_time_ms > 0.6:
        violations.append(f"Too broad (peak-to-trough {pt_time_ms:.3f} ms > 0.6 ms)")
    
    if pt_time_ms < 0.15:
        violations.append(f"Too narrow (peak-to-trough {pt_time_ms:.3f} ms < 0.15 ms)")
    
    if baseline_offset > 0.3:
        violations.append(f"Large baseline offset ({baseline_offset:.2f} of peak amplitude)")
    
    if not np.isnan(descent_time_ms) and descent_time_ms > 0.3:
        violations.append(f"Slow descent to negative extrema ({descent_time_ms:.3f} ms)")
    
    is_valid = len(violations) == 0
    
    return {
        'is_valid': is_valid,
        'violations': violations,
        'n_phases': n_phases,
        'peak_to_trough_ms': pt_time_ms,
        'baseline_offset': baseline_offset,
        'descent_time_ms': descent_time_ms,
        'peak_idx': peak_idx,
        'trough_idx': trough_idx,
    }


def detect_temporal_drift(spike_times: np.ndarray,
                         waveforms: np.ndarray,
                         n_epochs: int = 10,
                         fs: float = 30000) -> Dict[str, any]:
    """
    Detect temporal drift by analyzing waveform width changes over time.
    
    If waveform width changes by >20% across the recording, this suggests:
    - Electrode drift (neuron moving relative to electrode)
    - Multiple neurons with different waveforms mixed together
    
    Args:
        spike_times: (n_spikes,) spike times in seconds
        waveforms: (n_spikes, n_samples) waveform array
        n_epochs: Number of time bins to divide recording
        fs: Sampling frequency (Hz)
    
    Returns:
        Dict with:
            - drift_detected: bool (width changes >20%)
            - width_change_pct: float (max change percentage)
            - epoch_widths: List[float] (mean width per epoch in ms)
            - epoch_bounds: List[Tuple[float, float]] (time boundaries)
            - high_variability_epochs: List[int] (which epochs have high variability)
    """
    # Divide into epochs
    t_min, t_max = spike_times.min(), spike_times.max()
    epoch_edges = np.linspace(t_min, t_max, n_epochs + 1)
    
    epoch_widths = []
    epoch_stds = []
    epoch_bounds = []
    
    for i in range(n_epochs):
        t_start, t_end = epoch_edges[i], epoch_edges[i+1]
        mask = (spike_times >= t_start) & (spike_times < t_end)
        
        if mask.sum() < 10:  # Skip if too few spikes
            epoch_widths.append(np.nan)
            epoch_stds.append(np.nan)
            epoch_bounds.append((t_start, t_end))
            continue
        
        epoch_wf = waveforms[mask]
        mean_wf = epoch_wf.mean(axis=0)
        
        # Compute peak-to-trough width
        peak_idx = np.argmax(mean_wf)
        trough_idx = np.argmin(mean_wf)
        width_samples = abs(trough_idx - peak_idx)
        width_ms = width_samples / fs * 1000
        
        # Also compute std across waveforms in this epoch
        wf_std = epoch_wf.std(axis=0).mean()
        
        epoch_widths.append(width_ms)
        epoch_stds.append(wf_std)
        epoch_bounds.append((t_start, t_end))
    
    # Remove NaNs for analysis
    valid_widths = np.array([w for w in epoch_widths if not np.isnan(w)])
    valid_stds = np.array([s for s in epoch_stds if not np.isnan(s)])
    
    if len(valid_widths) < 2:
        return {
            'drift_detected': False,
            'width_change_pct': 0.0,
            'epoch_widths': epoch_widths,
            'epoch_bounds': epoch_bounds,
            'high_variability_epochs': [],
            'insufficient_data': True,
        }
    
    # Compute width change
    width_range = valid_widths.max() - valid_widths.min()
    width_change_pct = width_range / valid_widths.mean()
    
    # Detect drift
    drift_detected = width_change_pct > 0.2  # 20% threshold
    
    # Identify high variability epochs (std > mean std)
    mean_std = valid_stds.mean()
    high_variability_epochs = []
    for i, std in enumerate(epoch_stds):
        if not np.isnan(std) and std > 1.5 * mean_std:
            high_variability_epochs.append(i)
    
    return {
        'drift_detected': drift_detected,
        'width_change_pct': width_change_pct,
        'epoch_widths': epoch_widths,
        'epoch_bounds': epoch_bounds,
        'high_variability_epochs': high_variability_epochs,
        'mean_width_ms': valid_widths.mean(),
        'width_range_ms': width_range,
        'insufficient_data': False,
    }


def generate_action_log(actions: List[FilterAction], 
                       output_path: str = None) -> str:
    """
    Generate MATLAB-style action log from filter actions.
    
    Args:
        actions: List of FilterAction records
        output_path: Optional file path to save log
    
    Returns:
        Log string in MATLAB format
    """
    lines = ["Actions,Action Reasoning"]
    
    for action in actions:
        # MATLAB format: 'm 0 X', 'Reasoning text'
        line = f"{action.action},\"{action.reasoning}\""
        lines.append(line)
    
    log_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(log_str)
    
    return log_str


# =====================================================================
# Convenience function for full Phase 0 filtering
# =====================================================================

def phase0_automatic_filtering(assigns: np.ndarray,
                               size_threshold: int = 500,
                               log_path: str = None) -> Tuple[List[int], str]:
    """
    Complete Phase 0 automatic filtering workflow.
    
    Args:
        assigns: (n_spikes,) cluster assignments
        size_threshold: Minimum spike count
        log_path: Optional path to save action log
    
    Returns:
        kept_clusters: List of cluster IDs passing to Phase 1
        action_log: CSV-formatted log string
    """
    kept_clusters, actions = automatic_size_filter(assigns, size_threshold)
    
    action_log = generate_action_log(actions, log_path)
    
    # Print summary
    n_total = len(np.unique(assigns[assigns != 0]))
    n_discarded = len(actions)
    n_kept = len(kept_clusters)
    
    print(f"Phase 0 Automatic Filtering:")
    print(f"  Total clusters: {n_total}")
    print(f"  Discarded (< {size_threshold} spikes): {n_discarded} ({n_discarded/n_total*100:.1f}%)")
    print(f"  Kept for Phase 1: {n_kept} ({n_kept/n_total*100:.1f}%)")
    
    if log_path:
        print(f"  Action log saved: {log_path}")
    
    return kept_clusters, action_log
