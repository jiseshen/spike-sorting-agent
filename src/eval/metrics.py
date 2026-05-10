"""
Comprehensive metrics for spike sorting evaluation.

Two-tier system:
1. Quality Metrics (per-cluster): Using SpikeInterface quality metrics
   - ISI violations, firing rate, presence ratio, amplitude stats, SNR, etc.
   
2. Ground Truth Comparison (when available): Precision/Recall
   - Match curated clusters to ground truth
   - Compute TP/FP/FN, precision, recall, F1 score
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import spikeinterface as si
from spikeinterface.qualitymetrics import (
    compute_firing_rates,
    compute_isi_violations,
    compute_presence_ratios,
    compute_amplitude_cutoffs,
    compute_amplitude_medians,
    compute_amplitude_cv_metrics,
    compute_snrs,
)
from pathlib import Path


# =====================================================================
# 1. QUALITY METRICS (SpikeInterface-based)
# =====================================================================

def compute_quality_metrics(
    sorting: si.BaseSorting,
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    assigns: np.ndarray,
    sampling_frequency: float = 30000.0,
    refractory_period_ms: float = 2.0,
) -> pd.DataFrame:
    """
    Compute comprehensive quality metrics for each cluster.
    
    Uses lightweight implementations (no full SortingAnalyzer needed).
    
    Args:
        sorting: SpikeInterface sorting object
        waveforms: (n_spikes, n_samples) waveforms
        spike_times: (n_spikes,) spike times in seconds
        assigns: (n_spikes,) cluster assignments
        sampling_frequency: Hz
        refractory_period_ms: Refractory period in ms
        
    Returns:
        DataFrame with metrics per cluster
    """
    unit_ids = sorting.get_unit_ids()
    metrics_list = []
    
    for unit_id in unit_ids:
        # Get spike train
        spike_train = sorting.get_unit_spike_train(unit_id, segment_index=0)
        spike_times_unit = spike_train / sampling_frequency  # Convert to seconds
        
        # Get waveforms for this unit
        mask = assigns == unit_id
        waveforms_unit = waveforms[mask]
        
        # Basic stats
        n_spikes = len(spike_times_unit)
        duration = spike_times[-1] - spike_times[0] if len(spike_times) > 1 else 1.0
        firing_rate = n_spikes / duration
        
        # ISI violations
        if n_spikes >= 2:
            isis = np.diff(np.sort(spike_times_unit)) * 1000  # Convert to ms
            isi_violations = np.sum(isis < refractory_period_ms)
            isi_violations_rate = isi_violations / len(isis)
        else:
            isi_violations = 0
            isi_violations_rate = 0.0
        
        # Presence ratio (fraction of time with spikes)
        if n_spikes >= 2:
            bin_size = 60.0  # 60-second bins
            n_bins = int(np.ceil(duration / bin_size))
            spike_bins = (spike_times_unit / bin_size).astype(int)
            presence_ratio = len(np.unique(spike_bins)) / n_bins
        else:
            presence_ratio = 0.0
        
        # Amplitude stats
        if len(waveforms_unit) > 0:
            amplitudes = np.ptp(waveforms_unit, axis=1)  # Peak-to-trough per spike
            amplitude_median = np.median(amplitudes)
            amplitude_cv = np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0.0
            
            # Amplitude cutoff (Hill et al. 2011)
            # Estimates contamination by fitting histogram tail
            amplitude_cutoff = compute_amplitude_cutoff_single(amplitudes)
            
            # SNR (simplified - peak-to-trough / noise std)
            template = np.median(waveforms_unit, axis=0)
            noise_std = np.std(waveforms_unit - template)
            snr = np.ptp(template) / noise_std if noise_std > 0 else 0.0
        else:
            amplitude_median = 0.0
            amplitude_cv = 0.0
            amplitude_cutoff = 0.0
            snr = 0.0
        
        metrics_list.append({
            'cluster_id': int(unit_id),
            'n_spikes': n_spikes,
            'firing_rate': firing_rate,
            'isi_violations_rate': isi_violations_rate,
            'isi_violations_count': int(isi_violations),
            'presence_ratio': presence_ratio,
            'amplitude_median': amplitude_median,
            'amplitude_cv': amplitude_cv,
            'amplitude_cutoff': amplitude_cutoff,
            'snr': snr,
        })
    
    return pd.DataFrame(metrics_list)


def compute_amplitude_cutoff_single(amplitudes: np.ndarray, num_histogram_bins: int = 500) -> float:
    """
    Compute amplitude cutoff metric (Hill et al. 2011).
    
    Estimates fraction of spikes missing due to amplitude threshold.
    Based on fitting the tail of the amplitude distribution.
    
    Args:
        amplitudes: Peak-to-trough amplitudes
        num_histogram_bins: Number of histogram bins
        
    Returns:
        Amplitude cutoff value (lower is better, <0.1 is good)
    """
    if len(amplitudes) == 0:
        return 1.0
    
    # Create histogram
    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)
    
    # Find peak of histogram
    peak_idx = np.argmax(h)
    
    # Fit Gaussian to upper half (assumes contamination is low-amplitude tail)
    if peak_idx < len(h) - 1:
        upper_half = h[peak_idx:]
        upper_bins = b[peak_idx:-1]
        
        # Estimate cutoff as CDF of fitted distribution
        # Simplified: fraction of probability mass below threshold
        total_area = np.sum(h) * np.diff(b)[0]
        lower_area = np.sum(h[:peak_idx]) * np.diff(b)[0]
        cutoff = lower_area / total_area if total_area > 0 else 0.0
    else:
        cutoff = 0.0
    
    return cutoff


# =====================================================================
# 2. GROUND TRUTH COMPARISON (Precision/Recall)
# =====================================================================

def match_clusters_to_ground_truth(
    curated_sorting: si.BaseSorting,
    ground_truth_sorting: si.BaseSorting,
    curated_assigns: np.ndarray,
    gt_assigns: np.ndarray,
    sampling_frequency: float = 30000.0,
    delta_time_ms: float = 0.4,
) -> Tuple[pd.DataFrame, int]:
    """
    Match curated clusters to ground truth using direct spike assignment comparison.
    
    For each curated cluster, find the best-matching GT cluster and compute:
    - True Positives (TP): Spikes in both curated and GT
    - False Positives (FP): Spikes in curated but not in matched GT cluster
    - False Negatives (FN): Per-cluster FN (spikes in matched GT but not curated)
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN) [per-cluster recall for matched GT]
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    NOTE: Overall FN should be computed globally as (total_gt_spikes - total_tp)
          to account for unmatched GT clusters.
    
    Args:
        curated_sorting: Curated sorting result
        ground_truth_sorting: Ground truth sorting
        curated_assigns: (n_spikes,) curated cluster assignments
        gt_assigns: (n_spikes,) ground truth cluster assignments
        sampling_frequency: Hz
        delta_time_ms: Tolerance window for spike matching (ms) [not used, direct comparison]
        
    Returns:
        Tuple of (comparison_df, total_gt_spikes):
            - comparison_df: DataFrame with matching results per curated cluster
            - total_gt_spikes: Total number of GT spikes (for global FN calculation)
    """
    results = []
    
    # Count total GT spikes (excluding noise cluster 0 if present)
    gt_ids = ground_truth_sorting.get_unit_ids()
    total_gt_spikes = sum(np.sum(gt_assigns == gt_id) for gt_id in gt_ids)
    
    for curated_id in curated_sorting.get_unit_ids():
        # Get spikes in this curated cluster
        curated_mask = curated_assigns == curated_id
        n_curated = np.sum(curated_mask)
        
        # Find overlap with each GT cluster
        gt_ids = ground_truth_sorting.get_unit_ids()
        
        best_match_gt = None
        best_overlap = 0
        best_tp = 0
        best_fp = 0
        best_fn = 0
        
        for gt_id in gt_ids:
            # Get spikes in this GT cluster
            gt_mask = gt_assigns == gt_id
            n_gt = np.sum(gt_mask)
            
            # Compute overlap (spikes in both)
            overlap = np.sum(curated_mask & gt_mask)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_gt = gt_id
                
                # TP: spikes correctly assigned to this GT cluster
                best_tp = overlap
                
                # FP: spikes in curated cluster but not in this GT cluster
                # (includes spikes from other GT clusters + GT noise)
                best_fp = n_curated - overlap
                
                # FN: spikes in GT cluster but not in curated cluster
                best_fn = n_gt - overlap
        
        # If no overlap with any GT cluster, all are false positives
        if best_match_gt is None:
            best_tp = 0
            best_fp = n_curated
            best_fn = 0
        
        # Compute metrics
        precision = best_tp / (best_tp + best_fp) if (best_tp + best_fp) > 0 else 0.0
        recall = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'curated_id': int(curated_id),
            'matched_gt_id': int(best_match_gt) if best_match_gt is not None else None,
            'agreement': best_overlap / max(n_curated, np.sum(gt_assigns == best_match_gt)) if best_match_gt is not None else 0.0,
            'n_curated_spikes': n_curated,
            'n_gt_spikes': np.sum(gt_assigns == best_match_gt) if best_match_gt is not None else 0,
            'tp': best_tp,
            'fp': best_fp,
            'fn': best_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        })
    
    return pd.DataFrame(results), total_gt_spikes


def compute_overall_performance(comparison_df: pd.DataFrame, total_gt_spikes: int) -> Dict[str, float]:
    """
    Compute overall performance metrics across all clusters.
    
    CRITICAL: Uses global FN calculation to account for unmatched GT clusters.
    - TP: Sum of all correctly assigned spikes
    - FP: Sum of all incorrectly assigned spikes in curated clusters
    - FN: Total GT spikes - TP (includes unmatched GT clusters)
    
    This ensures that GT spikes in clusters with no curated match are counted as FN.
    
    Args:
        comparison_df: Output from match_clusters_to_ground_truth
        total_gt_spikes: Total number of GT spikes (from ground truth sorting)
        
    Returns:
        Dict with aggregate metrics
    """
    # Overall TP/FP (sum across curated clusters)
    total_tp = comparison_df['tp'].sum()
    total_fp = comparison_df['fp'].sum()
    
    # Global FN: All GT spikes that were NOT correctly recalled
    # This includes:
    # 1. Spikes in matched GT clusters that were missed (per-cluster FN)
    # 2. ALL spikes in GT clusters that have no curated match
    total_fn = total_gt_spikes - total_tp
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Mean metrics per cluster
    mean_precision = comparison_df['precision'].mean()
    mean_recall = comparison_df['recall'].mean()
    mean_f1 = comparison_df['f1_score'].mean()
    
    # Number of clusters
    n_curated = len(comparison_df)
    n_matched = comparison_df['matched_gt_id'].notna().sum()
    
    return {
        'n_curated_clusters': n_curated,
        'n_matched_clusters': n_matched,
        'total_gt_spikes': int(total_gt_spikes),
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1_score': mean_f1,
    }


# =====================================================================
# 3. COMBINED REPORT
# =====================================================================

def generate_full_evaluation_report(
    curated_sorting: si.BaseSorting,
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    assigns: np.ndarray,
    ground_truth_sorting: Optional[si.BaseSorting] = None,
    gt_assigns: Optional[np.ndarray] = None,
    sampling_frequency: float = 30000.0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        curated_sorting: Curated sorting result
        waveforms: (n_spikes, n_samples) waveforms
        spike_times: (n_spikes,) spike times in seconds
        assigns: (n_spikes,) curated cluster assignments
        ground_truth_sorting: Ground truth sorting (optional)
        gt_assigns: (n_spikes,) ground truth cluster assignments (optional)
        sampling_frequency: Hz
        output_dir: Directory to save CSV reports (optional)
        
    Returns:
        Dict with all metrics and DataFrames
    """
    report = {}
    
    # 1. Quality metrics
    print("Computing quality metrics...")
    quality_metrics = compute_quality_metrics(
        curated_sorting, waveforms, spike_times, assigns, sampling_frequency
    )
    report['quality_metrics'] = quality_metrics
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        quality_metrics.to_csv(output_dir / 'quality_metrics.csv', index=False)
        print(f"✓ Quality metrics saved to {output_dir / 'quality_metrics.csv'}")
    
    # 2. Ground truth comparison (if available)
    if ground_truth_sorting is not None and gt_assigns is not None:
        print("Computing ground truth comparison...")
        comparison_df, total_gt_spikes = match_clusters_to_ground_truth(
            curated_sorting, ground_truth_sorting, assigns, gt_assigns, sampling_frequency
        )
        report['ground_truth_comparison'] = comparison_df
        
        overall_perf = compute_overall_performance(comparison_df, total_gt_spikes)
        report['overall_performance'] = overall_perf
        
        if output_dir:
            comparison_df.to_csv(output_dir / 'ground_truth_comparison.csv', index=False)
            print(f"✓ Ground truth comparison saved to {output_dir / 'ground_truth_comparison.csv'}")
            
            # Save overall performance summary (TXT)
            with open(output_dir / 'overall_performance.txt', 'w') as f:
                f.write("OVERALL PERFORMANCE SUMMARY\n")
                f.write("="*50 + "\n\n")
                for key, value in overall_perf.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            print(f"✓ Overall performance saved to {output_dir / 'overall_performance.txt'}")
            
            # Save evaluation report (JSON) for aggregation
            import json
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            json_report = {
                'overall_performance': convert_to_native(overall_perf),
                'quality_metrics_summary': {
                    'n_clusters': len(quality_metrics),
                    'mean_firing_rate': float(quality_metrics['firing_rate'].mean()),
                    'mean_isi_violations': float(quality_metrics['isi_violations_rate'].mean()),
                    'mean_presence_ratio': float(quality_metrics['presence_ratio'].mean()),
                    'mean_amplitude_cv': float(quality_metrics['amplitude_cv'].mean()),
                    'mean_snr': float(quality_metrics['snr'].mean()),
                },
            }
            with open(output_dir / 'evaluation_report.json', 'w') as f:
                json.dump(json_report, f, indent=2)
            print(f"✓ Evaluation report saved to {output_dir / 'evaluation_report.json'}")
    else:
        print("⚠ No ground truth provided - skipping GT comparison")
        report['ground_truth_comparison'] = None
        report['overall_performance'] = None
    
    return report


def print_evaluation_summary(report: Dict[str, Any]):
    """Print evaluation summary to console."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Check if we have any metrics
    if report.get('quality_metrics') is None:
        print("\n⚠ No clusters remain - no metrics to display")
        print("="*70)
        return
    
    # Quality metrics summary
    qm = report['quality_metrics']
    print(f"\nQuality Metrics ({len(qm)} clusters):")
    print(f"  Mean firing rate: {qm['firing_rate'].mean():.2f} Hz")
    print(f"  Mean ISI violations: {qm['isi_violations_rate'].mean():.2%}")
    print(f"  Mean presence ratio: {qm['presence_ratio'].mean():.2%}")
    print(f"  Mean amplitude CV: {qm['amplitude_cv'].mean():.3f}")
    print(f"  Mean SNR: {qm['snr'].mean():.2f}")
    
    # Ground truth comparison summary
    if report['overall_performance'] is not None:
        perf = report['overall_performance']
        print(f"\nGround Truth Comparison:")
        print(f"  Curated clusters: {perf['n_curated_clusters']}")
        print(f"  Matched to GT: {perf['n_matched_clusters']}")
        print(f"  Overall Precision: {perf['overall_precision']:.2%}")
        print(f"  Overall Recall: {perf['overall_recall']:.2%}")
        print(f"  Overall F1 Score: {perf['overall_f1_score']:.3f}")
        print(f"  Mean Precision: {perf['mean_precision']:.2%}")
        print(f"  Mean Recall: {perf['mean_recall']:.2%}")
        print(f"  Mean F1 Score: {perf['mean_f1_score']:.3f}")
    
    print("="*70)
