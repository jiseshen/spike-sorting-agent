"""
Compute Human Curation (Ground Truth) quality metrics per channel and aggregate.

For each channel:
- Load data/{CH}_spikes.mat
- Build curated_sorting from GT assigns
- Run generate_full_evaluation_report to compute quality metrics
- Save into output/human_curation/{CH}

Then aggregate across channels and write:
- output/human_curation/aggregate_summary.csv
- output/human_curation/aggregate_performance.txt (means ± std)

Run: uv run python compute_human_curation.py
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import spikeinterface as si

from src.matlab_loader import convert_mat_to_sortings
from src.metrics import generate_full_evaluation_report

CHANNELS = ["CH3", "CH20", "CH30", "CH31"]
OUTPUT_BASE = Path("output/human_curation")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

all_results = []

for channel in CHANNELS:
    print("\n" + "="*80)
    print(f"HUMAN CURATION METRICS: {channel}")
    print("="*80)

    data_file = Path(f"data/{channel}_spikes.mat")
    if not data_file.exists():
        print(f"✗ Missing data file: {data_file}")
        continue

    try:
        sorting, sorting_tree, meta = convert_mat_to_sortings(str(data_file))
    except Exception as e:
        print(f"✗ Failed to load {data_file}: {e}")
        continue

    gt_assigns = meta.get("curation_assigns")
    waveforms = meta["waveforms"]
    spike_times_all = meta["spiketimes"]
    Fs = meta["Fs"]

    if gt_assigns is None:
        print("✗ No GT assigns found in meta['curation_assigns']")
        continue

    # Build curated_sorting from GT assigns
    spike_frames = (spike_times_all * Fs).astype(np.int64)
    gt_cluster_ids = np.unique(gt_assigns[gt_assigns > 0])
    if len(gt_cluster_ids) == 0:
        print("✗ GT has zero clusters")
        continue

    curated_sorting = si.NumpySorting.from_unit_dict(
        {int(cid): spike_frames[gt_assigns == cid] for cid in gt_cluster_ids},
        sampling_frequency=Fs
    )

    # Ground truth sorting is the same (for completeness)
    gt_sorting = curated_sorting

    # Output dir
    ch_out = OUTPUT_BASE / channel
    ch_out.mkdir(parents=True, exist_ok=True)

    # Run evaluation to compute quality metrics (and overall = 1.0)
    report = generate_full_evaluation_report(
        curated_sorting=curated_sorting,
        waveforms=waveforms,
        spike_times=spike_times_all,
        assigns=gt_assigns,
        ground_truth_sorting=gt_sorting,
        gt_assigns=gt_assigns,
        sampling_frequency=Fs,
        output_dir=ch_out,
    )

    perf = report.get('overall_performance') or {}
    qm_path = ch_out / 'quality_metrics.csv'
    if not qm_path.exists():
        print("⚠ quality_metrics.csv not found; skipping channel aggregation")
        continue

    qdf = pd.read_csv(qm_path)

    # Summarize per-channel quality metrics
    perf_summary = {
        'channel': channel,
        'n_final_clusters': len(qdf),
        'overall_precision': 1.0,
        'overall_recall': 1.0,
        'overall_f1_score': 1.0,
        'mean_firing_rate': float(qdf['firing_rate'].mean()),
        'mean_isi_violations': float(qdf['isi_violations_rate'].mean()),
        'mean_presence_ratio': float(qdf['presence_ratio'].mean()),
        'mean_amplitude_cv': float(qdf['amplitude_cv'].mean()),
        'mean_snr': float(qdf['snr'].mean()),
    }
    all_results.append(perf_summary)

# Aggregate
if not all_results:
    print("\n⚠ No channels aggregated; aborting")
else:
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_BASE / 'aggregate_summary.csv', index=False)
    print(f"\n✓ Saved per-channel summary: {OUTPUT_BASE / 'aggregate_summary.csv'}")

    # Write aggregate_performance.txt in the same format used by plotting scripts
    report_path = OUTPUT_BASE / 'aggregate_performance.txt'
    with open(report_path, 'w') as f:
        f.write("AGGREGATE PERFORMANCE ACROSS ALL CHANNELS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Channels processed: {len(df)}\n")
        f.write("Model: human_curation\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-"*60 + "\n")
        f.write("MEAN METRICS (Ground Truth Comparison)\n")
        f.write("-"*60 + "\n")
        for metric in ['overall_precision','overall_recall','overall_f1_score']:
            f.write(f"{metric}: {df[metric].mean():.4f} ± {df[metric].std():.4f}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("MEAN METRICS (Quality Metrics)\n")
        f.write("-"*60 + "\n")
        for metric in ['mean_firing_rate','mean_isi_violations','mean_presence_ratio','mean_amplitude_cv','mean_snr']:
            f.write(f"{metric}: {df[metric].mean():.4f} ± {df[metric].std():.4f}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("PER-CHANNEL SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(df.to_string(index=False))

    print(f"✓ Saved aggregate performance: {report_path}")
