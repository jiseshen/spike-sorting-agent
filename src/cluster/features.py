"""Cluster feature extraction (Phase 0).

Provides a lightweight, memory-safe feature computation module used by
later Agent phases (split / merge decisions).

Features per cluster:
- n_spikes
- firing_rate_hz
- isi_violation_rate (% of ISIs < refractory)
- isi_score (MATLAB-style quality score)
- mean_waveform_correlation (consistency proxy)
- waveform_amplitude_mean / std / cv
- waveform_peak_to_trough_mean
- overcluster_count (composition granularity)

Design principles:
- Accept raw meta dict from matlab_loader + current assigns
- Subsample waveforms for large clusters automatically (configurable)
- Avoid full O(N^2) correlations by sampling and correlating to mean

Correlation usage rationale:
- Low mean correlation (<0.70) -> waveform heterogeneity -> candidate for split
- Moderate correlation (0.70-0.85) + high ISI violations -> stronger split signal
- High correlation (>0.90) across small clusters -> candidate for merge if ISI clean

Cluster 0 (noise) handling:
- Excluded from feature table by default unless include_noise=True.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


def compute_isi_violation_rate(spike_times: np.ndarray, 
                               tref: float = 0.002) -> Tuple[float, int, int]:
    """
    Compute ISI violation rate.
    
    Args:
        spike_times: Spike times in seconds
        tref: Refractory period in seconds
        
    Returns:
        (violation_rate, n_violations, n_isis)
    """
    if len(spike_times) < 2:
        return 0.0, 0, 0
    
    isis = np.diff(np.sort(spike_times))
    n_violations = int(np.sum(isis < tref))
    n_isis = len(isis)
    violation_rate = n_violations / n_isis if n_isis > 0 else 0.0
    
    return violation_rate, n_violations, n_isis


def compute_isi_quality(spike_times: np.ndarray,
                        tref: float = 0.002,
                        tmax: float = 0.050) -> Tuple[float, int, int]:
    """
    Compute MATLAB-style ISI quality score.
    
    Args:
        spike_times: Spike times in seconds
        tref: Refractory period threshold
        tmax: Maximum ISI to consider
        
    Returns:
        (quality_score, n_violations, n_isis_in_range)
    """
    if len(spike_times) < 2:
        return 1.0, 0, 0
    
    isis = np.diff(np.sort(spike_times))
    in_range = (isis > 0) & (isis <= tmax)
    isis_in_range = isis[in_range]
    
    if len(isis_in_range) == 0:
        return 1.0, 0, 0
    
    n_violations = int(np.sum(isis_in_range < tref))
    n_total = len(isis_in_range)
    
    # Quality score: 1 - (violations / total)
    quality = 1.0 - (n_violations / n_total)
    
    return quality, n_violations, n_total


@dataclass
class ClusterFeatureRow:
    cluster_id: int
    n_spikes: int
    firing_rate_hz: float
    isi_violation_rate: float
    isi_score: float
    mean_waveform_correlation: float
    amplitude_mean: float
    amplitude_std: float
    amplitude_cv: float
    peak_to_trough_mean: float
    overcluster_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'n_spikes': self.n_spikes,
            'firing_rate_hz': round(self.firing_rate_hz, 3),
            'isi_violation_rate': round(self.isi_violation_rate, 4),
            'isi_score': round(self.isi_score, 3),
            'mean_waveform_correlation': round(self.mean_waveform_correlation, 3),
            'amplitude_mean': round(self.amplitude_mean, 3),
            'amplitude_std': round(self.amplitude_std, 3),
            'amplitude_cv': round(self.amplitude_cv, 3),
            'peak_to_trough_mean': round(self.peak_to_trough_mean, 3),
            'overcluster_count': self.overcluster_count,
        }


class ClusterFeatures:
    def __init__(self,
                 meta: Dict[str, Any],
                 assigns: np.ndarray,
                 refractory_period: Optional[float] = None,
                 max_waveforms_for_corr: int = 5000,
                 sample_waveforms_for_corr: int = 200,
                 include_noise: bool = False):
        self.meta = meta
        self.assigns = np.asarray(assigns).flatten()
        self.spike_times = np.asarray(meta['spiketimes']).flatten()
        self.waveforms = np.asarray(meta['waveforms'])  # (n_spikes, n_samples)
        self.overclusters = np.asarray(meta['overcluster_assigns']).flatten()
        self.fs = float(meta['Fs'])
        self.refractory = refractory_period if refractory_period is not None else float(meta.get('refractory_period', 0.002))
        self.max_waveforms_for_corr = max_waveforms_for_corr
        self.sample_waveforms_for_corr = sample_waveforms_for_corr
        self.include_noise = include_noise

    def _compute_waveform_stats(self, wf: np.ndarray) -> Dict[str, float]:
        # Amplitude per spike (peak - trough)
        peak = wf.max(axis=1)
        trough = wf.min(axis=1)
        amplitude = peak - trough
        return {
            'amplitude_mean': float(amplitude.mean()),
            'amplitude_std': float(amplitude.std(ddof=1)) if len(amplitude) > 1 else 0.0,
            'amplitude_cv': float(amplitude.std(ddof=1) / amplitude.mean()) if amplitude.mean() > 0 and len(amplitude) > 1 else 0.0,
            'peak_to_trough_mean': float((peak - trough).mean()),
        }

    def _compute_correlation(self, wf: np.ndarray) -> float:
        if wf.shape[0] == 0:
            return 0.0
        if wf.shape[0] > self.max_waveforms_for_corr:
            idx = np.random.choice(wf.shape[0], self.max_waveforms_for_corr, replace=False)
            wf = wf[idx]
        mean_wf = wf.mean(axis=0)
        # Sample subset for correlation to mean
        sample_n = min(self.sample_waveforms_for_corr, wf.shape[0])
        sample_idx = np.random.choice(wf.shape[0], sample_n, replace=False)
        corrs = []
        for i in sample_idx:
            a = wf[i]
            # Pearson corr manually to avoid building huge matrix
            va = a - a.mean()
            vm = mean_wf - mean_wf.mean()
            denom = np.linalg.norm(va) * np.linalg.norm(vm)
            corrs.append(float(va.dot(vm) / denom) if denom > 0 else 0.0)
        return float(np.mean(corrs)) if corrs else 0.0

    def compute(self) -> List[ClusterFeatureRow]:
        rows: List[ClusterFeatureRow] = []
        cluster_ids = np.unique(self.assigns)
        if not self.include_noise and 0 in cluster_ids and len(cluster_ids) > 1:
            cluster_ids = cluster_ids[cluster_ids != 0]
        cluster_ids = cluster_ids.tolist()

        for cid in cluster_ids:
            mask = (self.assigns == cid)
            n_spikes = int(mask.sum())
            if n_spikes == 0:
                continue
            spike_times_c = self.spike_times[mask]
            spike_times_c.sort()
            duration = spike_times_c[-1] - spike_times_c[0] if n_spikes > 1 else 0.0
            firing_rate = n_spikes / duration if duration > 0 else 0.0

            # ISI metrics
            isi_score, _, _ = compute_isi_quality(spike_times_c, tref=self.refractory, tmax=0.050)
            violation_rate, _, _ = compute_isi_violation_rate(spike_times_c, tref=self.refractory)

            wf_cluster = self.waveforms[mask]
            corr = self._compute_correlation(wf_cluster)
            wf_stats = self._compute_waveform_stats(wf_cluster)
            overcluster_count = int(np.unique(self.overclusters[mask]).size)

            row = ClusterFeatureRow(
                cluster_id=int(cid),
                n_spikes=n_spikes,
                firing_rate_hz=firing_rate,
                isi_violation_rate=violation_rate,
                isi_score=float(isi_score),
                mean_waveform_correlation=corr,
                amplitude_mean=wf_stats['amplitude_mean'],
                amplitude_std=wf_stats['amplitude_std'],
                amplitude_cv=wf_stats['amplitude_cv'],
                peak_to_trough_mean=wf_stats['peak_to_trough_mean'],
                overcluster_count=overcluster_count,
            )
            rows.append(row)
        return rows

    @staticmethod
    def to_table(rows: List[ClusterFeatureRow]) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in rows]

    @staticmethod
    def summarize(rows: List[ClusterFeatureRow]) -> Dict[str, Any]:
        return {
            'n_clusters': len(rows),
            'total_spikes': int(sum(r.n_spikes for r in rows)),
            'mean_violation_rate': float(np.mean([r.isi_violation_rate for r in rows])) if rows else 0.0,
            'mean_waveform_corr': float(np.mean([r.mean_waveform_correlation for r in rows])) if rows else 0.0,
            'mean_amplitude_cv': float(np.mean([r.amplitude_cv for r in rows])) if rows else 0.0,
        }
