"""
Aggregate baseline (pre-curation) results across all channels.
"""

from pathlib import Path
import json
import pandas as pd

OUTPUT_DIR = Path("output/before_curation")
CHANNELS = ["CH3", "CH20", "CH30", "CH31"]

print("="*80)
print("AGGREGATING BASELINE (PRE-CURATION) RESULTS")
print("="*80)

all_results = []

for channel in CHANNELS:
    channel_dir = OUTPUT_DIR / channel
    eval_report_path = channel_dir / "evaluation_report.json"
    
    if not eval_report_path.exists():
        print(f"⚠ {channel}: No evaluation report found")
        continue
    
    # Load JSON report
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    
    perf = report['overall_performance']
    quality = report['quality_metrics_summary']
    
    result = {
        'channel': channel,
        'n_hierarchy_clusters': perf['n_curated_clusters'],
        'n_matched_to_gt': perf['n_matched_clusters'],
        'total_gt_spikes': perf['total_gt_spikes'],
        'total_tp': perf['total_tp'],
        'total_fp': perf['total_fp'],
        'total_fn': perf['total_fn'],
        'overall_precision': perf['overall_precision'],
        'overall_recall': perf['overall_recall'],
        'overall_f1_score': perf['overall_f1_score'],
        'mean_firing_rate': quality['mean_firing_rate'],
        'mean_isi_violations': quality['mean_isi_violations'],
        'mean_snr': quality['mean_snr'],
    }
    
    all_results.append(result)
    print(f"✓ {channel}: P={perf['overall_precision']:.4f}, R={perf['overall_recall']:.4f}, F1={perf['overall_f1_score']:.4f}")

# Create DataFrame
df = pd.DataFrame(all_results)

# Compute aggregate from total TP/FP/FN
total_tp = df['total_tp'].sum()
total_fp = df['total_fp'].sum()
total_fn = df['total_fn'].sum()
total_gt = df['total_gt_spikes'].sum()

agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
agg_f1 = 2 * (agg_precision * agg_recall) / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0.0

print("\n" + "="*80)
print("AGGREGATE BASELINE RESULTS")
print("="*80)
print(f"\nTotal hierarchy clusters: {df['n_hierarchy_clusters'].sum()}")
print(f"Total GT spikes: {total_gt}")
print(f"Total TP: {total_tp}")
print(f"Total FP: {total_fp}")
print(f"Total FN: {total_fn}")
print(f"\nAggregate Performance (from total TP/FP/FN):")
print(f"  Precision: {agg_precision:.4f}")
print(f"  Recall:    {agg_recall:.4f}")
print(f"  F1 Score:  {agg_f1:.4f}")

print(f"\nPer-channel means:")
print(f"  Precision: {df['overall_precision'].mean():.4f} ± {df['overall_precision'].std():.4f}")
print(f"  Recall:    {df['overall_recall'].mean():.4f} ± {df['overall_recall'].std():.4f}")
print(f"  F1 Score:  {df['overall_f1_score'].mean():.4f} ± {df['overall_f1_score'].std():.4f}")

print(f"\nQuality Metrics:")
print(f"  Mean firing rate:    {df['mean_firing_rate'].mean():.2f} ± {df['mean_firing_rate'].std():.2f} Hz")
print(f"  Mean ISI violations: {df['mean_isi_violations'].mean():.2%} ± {df['mean_isi_violations'].std():.2%}")
print(f"  Mean SNR:            {df['mean_snr'].mean():.2f} ± {df['mean_snr'].std():.2f}")

# Save summary
df.to_csv(OUTPUT_DIR / "aggregate_summary.csv", index=False)
print(f"\n✓ Summary saved to {OUTPUT_DIR / 'aggregate_summary.csv'}")

# Save detailed report
with open(OUTPUT_DIR / "aggregate_performance.txt", 'w') as f:
    f.write("BASELINE HIERARCHY CLUSTERING PERFORMANCE\n")
    f.write("(Before any VLM curation)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total hierarchy clusters: {df['n_hierarchy_clusters'].sum()}\n")
    f.write(f"Total GT spikes: {total_gt}\n\n")
    
    f.write("AGGREGATE PERFORMANCE (from total TP/FP/FN)\n")
    f.write("-"*60 + "\n")
    f.write(f"Total TP: {total_tp}\n")
    f.write(f"Total FP: {total_fp}\n")
    f.write(f"Total FN: {total_fn}\n")
    f.write(f"Aggregate Precision: {agg_precision:.4f}\n")
    f.write(f"Aggregate Recall:    {agg_recall:.4f}\n")
    f.write(f"Aggregate F1 Score:  {agg_f1:.4f}\n\n")
    
    f.write("PER-CHANNEL MEANS\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean Precision: {df['overall_precision'].mean():.4f} ± {df['overall_precision'].std():.4f}\n")
    f.write(f"Mean Recall:    {df['overall_recall'].mean():.4f} ± {df['overall_recall'].std():.4f}\n")
    f.write(f"Mean F1 Score:  {df['overall_f1_score'].mean():.4f} ± {df['overall_f1_score'].std():.4f}\n\n")
    
    f.write("QUALITY METRICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Mean firing rate:    {df['mean_firing_rate'].mean():.2f} ± {df['mean_firing_rate'].std():.2f} Hz\n")
    f.write(f"Mean ISI violations: {df['mean_isi_violations'].mean():.4f} ± {df['mean_isi_violations'].std():.4f}\n")
    f.write(f"Mean SNR:            {df['mean_snr'].mean():.2f} ± {df['mean_snr'].std():.2f}\n\n")
    
    f.write("PER-CHANNEL DETAILS\n")
    f.write("-"*60 + "\n")
    f.write(df[['channel', 'n_hierarchy_clusters', 'overall_precision', 'overall_recall', 'overall_f1_score']].to_string(index=False))

print(f"✓ Report saved to {OUTPUT_DIR / 'aggregate_performance.txt'}")

print("\n" + "="*80)
print("BASELINE AGGREGATION COMPLETE")
print("="*80)
