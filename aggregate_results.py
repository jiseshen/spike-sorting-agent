"""
Aggregate results from all channels without re-running the pipeline.

Reads existing evaluation reports and computes aggregate statistics.
If evaluation reports are missing but final_assigns exist, computes them on-the-fly.

Usage: uv run aggregate_results.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import spikeinterface as si

from src.matlab_loader import convert_mat_to_sortings
from src.metrics import generate_full_evaluation_report

# =====================================================================
# CONFIGURATION
# =====================================================================

MODEL = "gpt-5.1"  # Must match the model used in run_all_channels.py
OUTPUT_BASE = Path(f"output/before_curation")
CHANNELS = ["CH3", "CH20", "CH30", "CH31"]

# =====================================================================
# AGGREGATE EXISTING RESULTS
# =====================================================================

print("="*80)
print("AGGREGATING EXISTING RESULTS")
print("="*80)
print(f"Model: {MODEL}")
print(f"Output directory: {OUTPUT_BASE}")
print(f"Channels: {', '.join(CHANNELS)}")
print("="*80)

all_results = []
channel_stats = []

for channel in CHANNELS:
    channel_output = OUTPUT_BASE / channel
    
    # Check if channel output exists
    if not channel_output.exists():
        print(f"\n⚠ {channel}: No output directory found, skipping")
        continue
    
    print(f"\n→ Reading {channel} results...")
    
    # Try reading JSON first, fall back to TXT, then recompute if needed
    eval_report_path = channel_output / "evaluation_report.json"
    perf = None
    quality_metrics = None
    
    if eval_report_path.exists():
        # Read from JSON (new format)
        try:
            with open(eval_report_path, 'r') as f:
                report = json.load(f)
            
            if report.get('overall_performance') is not None:
                perf = report['overall_performance'].copy()
                print(f"  ✓ Loaded from JSON")
        except Exception as e:
            print(f"  ⚠ Error reading JSON: {e}")
    
    if perf is None:
        # Fall back to parsing TXT file (old format)
        txt_path = channel_output / "overall_performance.txt"
        if txt_path.exists():
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                perf = {}
                for line in lines:
                    line = line.strip()
                    if ':' in line and not line.startswith('='):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse numeric values
                        try:
                            if '.' in value:
                                perf[key] = float(value)
                            else:
                                perf[key] = int(value)
                        except ValueError:
                            perf[key] = value
                
                print(f"  ✓ Loaded from TXT (fallback)")
            except Exception as e:
                print(f"  ⚠ Error reading TXT: {e}")
        else:
            print(f"  ⚠ No evaluation report found, trying to recompute...")
            
            # Last resort: recompute from final_assigns if available
            final_assigns_path = channel_output / "final_assigns.npy"
            if final_assigns_path.exists():
                try:
                    # Load data
                    data_file = Path(f"data/{channel}_spikes.mat")
                    if not data_file.exists():
                        print(f"  ✗ Data file not found: {data_file}")
                        continue
                    
                    print(f"  → Recomputing evaluation from final_assigns...")
                    sorting, sorting_tree, meta = convert_mat_to_sortings(str(data_file))
                    waveforms = meta["waveforms"]
                    spike_times_all = meta["spiketimes"]
                    Fs = meta["Fs"]
                    gt_assigns = meta.get("curation_assigns")
                    
                    # Load final assigns
                    final_assigns = np.load(final_assigns_path)
                    
                    # Get final cluster IDs
                    final_cluster_ids = np.unique(final_assigns[final_assigns > 0])
                    
                    if len(final_cluster_ids) == 0:
                        print(f"  ✗ No valid clusters in final_assigns")
                        continue
                    
                    # Create final sorting
                    spike_frames = (spike_times_all * Fs).astype(np.int64)
                    final_sorting = si.NumpySorting.from_unit_dict(
                        {int(cid): spike_frames[final_assigns == cid] for cid in final_cluster_ids},
                        sampling_frequency=Fs
                    )
                    
                    # Create ground truth sorting
                    gt_sorting = None
                    if gt_assigns is not None:
                        gt_cluster_ids = np.unique(gt_assigns[gt_assigns > 0])
                        gt_sorting = si.NumpySorting.from_unit_dict(
                            {int(cid): spike_frames[gt_assigns == cid] for cid in gt_cluster_ids},
                            sampling_frequency=Fs
                        )
                    
                    # Run evaluation
                    report = generate_full_evaluation_report(
                        curated_sorting=final_sorting,
                        waveforms=waveforms,
                        spike_times=spike_times_all,
                        assigns=final_assigns,
                        ground_truth_sorting=gt_sorting,
                        gt_assigns=gt_assigns,
                        sampling_frequency=Fs,
                        output_dir=channel_output,
                    )
                    
                    if report['overall_performance'] is not None:
                        perf = report['overall_performance'].copy()
                        print(f"  ✓ Recomputed evaluation successfully")
                    else:
                        print(f"  ✗ Recomputation failed (no GT?)")
                        continue
                    
                except Exception as e:
                    print(f"  ✗ Error recomputing evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                print(f"  ✗ No final_assigns.npy found, cannot recompute")
                continue
    
    # Read quality metrics if available
    quality_path = channel_output / "quality_metrics.csv"
    if quality_path.exists():
        quality_metrics = pd.read_csv(quality_path)
    
    # Extract performance metrics
    if perf:
        perf['channel'] = channel
        
        # Add quality metrics averages
        if quality_metrics is not None:
            perf['n_final_clusters'] = len(quality_metrics)
            perf['mean_firing_rate'] = float(quality_metrics['firing_rate'].mean())
            perf['mean_isi_violations'] = float(quality_metrics['isi_violations_rate'].mean())
            perf['mean_presence_ratio'] = float(quality_metrics['presence_ratio'].mean())
            perf['mean_amplitude_cv'] = float(quality_metrics['amplitude_cv'].mean())
            perf['mean_snr'] = float(quality_metrics['snr'].mean())
        
        # Set elapsed time to None (not available from saved reports)
        if 'elapsed_time_s' not in perf:
            perf['elapsed_time_s'] = None
        
        all_results.append(perf)
        print(f"  ✓ Loaded metrics: Precision={perf.get('overall_precision', 0):.4f}, "
              f"Recall={perf.get('overall_recall', 0):.4f}, "
              f"F1={perf.get('overall_f1_score', 0):.4f}")
    else:
        print(f"  ✗ No performance metrics found")

if not all_results:
    print("\n⚠ No results found to aggregate")
    sys.exit(1)

# Create summary DataFrame
summary_df = pd.DataFrame(all_results)

# Calculate mean metrics - only overall_* for GT comparison, plus quality metrics
gt_metrics = [
    'overall_precision', 'overall_recall', 'overall_f1_score',
]

quality_metrics_list = [
    'mean_firing_rate', 'mean_isi_violations', 'mean_presence_ratio',
    'mean_amplitude_cv', 'mean_snr'
]

print("\n" + "="*80)
print("AGGREGATE RESULTS")
print("="*80)

print("\n" + "-"*80)
print("PER-CHANNEL RESULTS")
print("-"*80)
# Select columns to display
display_cols = ['channel', 'n_final_clusters'] + gt_metrics + quality_metrics_list
available_cols = [c for c in display_cols if c in summary_df.columns]
print(summary_df[available_cols].to_string(index=False))

print("\n" + "-"*80)
print("AGGREGATE STATISTICS (Ground Truth Comparison)")
print("-"*80)

# Compute aggregate from total TP/FP/FN across all channels (correct method)
if 'total_tp' in summary_df.columns and 'total_fp' in summary_df.columns and 'total_fn' in summary_df.columns:
    total_tp_all = summary_df['total_tp'].sum()
    total_fp_all = summary_df['total_fp'].sum()
    total_fn_all = summary_df['total_fn'].sum()
    
    agg_precision = total_tp_all / (total_tp_all + total_fp_all) if (total_tp_all + total_fp_all) > 0 else 0.0
    agg_recall = total_tp_all / (total_tp_all + total_fn_all) if (total_tp_all + total_fn_all) > 0 else 0.0
    agg_f1 = 2 * (agg_precision * agg_recall) / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0.0
    
    print(f"{'aggregate_precision':25s}: {agg_precision:.4f} (from total TP/FP)")
    print(f"{'aggregate_recall':25s}: {agg_recall:.4f} (from total TP/FN)")
    print(f"{'aggregate_f1_score':25s}: {agg_f1:.4f} (from total TP/FP/FN)")
    print()

# Also show per-channel means for reference
for metric in gt_metrics:
    if metric in summary_df.columns:
        mean_val = summary_df[metric].mean()
        std_val = summary_df[metric].std()
        print(f"{metric:25s}: {mean_val:.4f} ± {std_val:.4f} (per-channel mean)")

print("\n" + "-"*80)
print("AGGREGATE STATISTICS (Quality Metrics)")
print("-"*80)

for metric in quality_metrics_list:
    if metric in summary_df.columns:
        mean_val = summary_df[metric].mean()
        std_val = summary_df[metric].std()
        print(f"{metric:25s}: {mean_val:.4f} ± {std_val:.4f}")

# Save summary
summary_path = OUTPUT_BASE / "aggregate_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary saved to {summary_path}")

# Create aggregated performance report
report_path = OUTPUT_BASE / "aggregate_performance.txt"
with open(report_path, 'w') as f:
    f.write("AGGREGATE PERFORMANCE ACROSS ALL CHANNELS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Channels processed: {len(all_results)}\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("-"*60 + "\n")
    f.write("MEAN METRICS (Ground Truth Comparison)\n")
    f.write("-"*60 + "\n")
    for metric in gt_metrics:
        if metric in summary_df.columns:
            mean_val = summary_df[metric].mean()
            std_val = summary_df[metric].std()
            f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    f.write("\n" + "-"*60 + "\n")
    f.write("MEAN METRICS (Quality Metrics)\n")
    f.write("-"*60 + "\n")
    for metric in quality_metrics_list:
        if metric in summary_df.columns:
            mean_val = summary_df[metric].mean()
            std_val = summary_df[metric].std()
            f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    f.write("\n" + "-"*60 + "\n")
    f.write("PER-CHANNEL SUMMARY\n")
    f.write("-"*60 + "\n")
    display_cols = ['channel', 'n_final_clusters'] + gt_metrics + quality_metrics_list
    available_cols = [c for c in display_cols if c in summary_df.columns]
    f.write(summary_df[available_cols].to_string(index=False))

print(f"✓ Aggregate report saved to {report_path}")

print("\n" + "="*80)
print("AGGREGATION COMPLETE")
print("="*80)
print(f"Results saved to: {OUTPUT_BASE.absolute()}")
print("="*80)
