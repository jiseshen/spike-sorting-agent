"""
Generate professional multi-panel comparison figure for spike sorting methods.

Methods compared:
- baseline (before curation)
- gpt-4.1 (main_gpt-4.1)
- gpt-5.1 (main_gpt-5.1)
- ablation (ablation_no_metrics)

Metrics used (from evaluation_report.json overall + quality metrics section):
- overall_f1_score
- overall_precision
- overall_recall
- mean_firing_rate
- mean_isi_violation
- mean_presence_ratio
- mean_amplitude_cv
- mean_snr

Figure panels:
1. Bar chart with error bars (Precision / Recall / F1 across methods)
2. Radar (spider) plot (normalized multi-metric profile) — invert "bad" metrics (isi_violation, amplitude_cv)
3. Heatmap (methods x metrics, z-score normalized across methods)
4. Scatter: SNR vs Amplitude CV (per-channel points colored by method) with marginal KDE

Outputs:
- PNG: output/visualizations/comparison/comparison_multimetrics.png
- CSV: output/visualizations/comparison/metrics_aggregated.csv
"""
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Attempt to use scienceplots for nicer style if available
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science','no-latex','grid'])
except Exception:
    plt.style.use('seaborn-v0_8-whitegrid')

METHOD_SPECS = [
    ("baseline", "output/before_curation"),
    ("gpt-4.1", "output/main_gpt-4.1"),
    ("gpt-5.1", "output/main_gpt-5.1"),
    ("ablation", "output/ablation_no_metrics"),
]

METRICS = [
    "overall_f1_score",
    "overall_precision",
    "overall_recall",
    "mean_firing_rate",
    "mean_isi_violation",
    "mean_presence_ratio",
    "mean_amplitude_cv",
    "mean_snr",
]

# Metrics where lower is better → will invert for radar normalization
INVERT_METRICS = {"mean_isi_violation", "mean_amplitude_cv"}

COLORMAP = {
    "baseline": "#888888",
    "gpt-4.1": "#1f77b4",
    "gpt-5.1": "#d62728",
    "ablation": "#9467bd",
}

def load_method_channel_metrics(base_dir: str, method: str) -> pd.DataFrame:
    rows = []
    base = Path(base_dir)
    if not base.exists():
        return pd.DataFrame(columns=['channel'] + METRICS)
    for ch_dir in sorted(base.glob('CH*')):
        report_path = ch_dir / 'evaluation_report.json'
        if not report_path.exists():
            continue
        with open(report_path, 'r') as f:
            data = json.load(f)
        overall = data.get('overall_performance', {}) or {}
        quality = data.get('quality_metrics', {}) or {}
        row = {
            'method': method,
            'channel': ch_dir.name,
            'overall_f1_score': overall.get('overall_f1_score', np.nan),
            'overall_precision': overall.get('overall_precision', np.nan),
            'overall_recall': overall.get('overall_recall', np.nan),
            'mean_firing_rate': quality.get('mean_firing_rate', np.nan),
            'mean_isi_violation': quality.get('mean_isi_violation', np.nan),
            'mean_presence_ratio': quality.get('mean_presence_ratio', np.nan),
            'mean_amplitude_cv': quality.get('mean_amplitude_cv', np.nan),
            'mean_snr': quality.get('mean_snr', np.nan),
        }
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('method')[METRICS].agg(['mean','std'])
    # Flatten columns
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg.reset_index(inplace=True)
    return agg

def prepare_radar_values(agg_mean: pd.DataFrame) -> pd.DataFrame:
    # Normalize each metric to [0,1] across methods; invert if needed
    norm_rows = []
    for _, row in agg_mean.iterrows():
        norm_row = {'method': row['method']}
        for metric in METRICS:
            vals = agg_mean[metric].values
            min_v, max_v = np.nanmin(vals), np.nanmax(vals)
            if max_v - min_v < 1e-9:
                norm = 0.5
            else:
                raw = row[metric]
                if metric in INVERT_METRICS:
                    # Lower better → invert scale
                    raw = max_v - (row[metric] - min_v)
                    min_v, max_v = min_v, max_v  # unchanged
                    # Recompute scale after inversion
                    vals_inv = max_v - (vals - min_v)
                    min_inv, max_inv = np.nanmin(vals_inv), np.nanmax(vals_inv)
                    norm = (raw - min_inv) / (max_inv - min_inv + 1e-9)
                else:
                    norm = (raw - min_v) / (max_v - min_v)
            norm_row[metric] = norm
        norm_rows.append(norm_row)
    return pd.DataFrame(norm_rows)

def plot_comparison(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_metrics(df)
    # Extract mean columns
    mean_cols = [m for m in agg.columns if m.endswith('_mean')]
    std_cols = [m for m in agg.columns if m.endswith('_std')]
    # Build mean-only frame
    agg_mean = agg[['method'] + [c for c in mean_cols]]
    # Rename mean columns back to metric names
    rename_map = {c: c.replace('_mean','') for c in mean_cols}
    agg_mean.rename(columns=rename_map, inplace=True)

    radar_df = prepare_radar_values(agg_mean)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    # Panel 1: Bars for Precision / Recall / F1
    ax1 = fig.add_subplot(gs[0,0])
    pr_metrics = ['overall_precision','overall_recall','overall_f1_score']
    bar_data = agg_mean.melt(id_vars='method', value_vars=pr_metrics, var_name='metric', value_name='value')
    sns.barplot(data=bar_data, x='metric', y='value', hue='method', palette=COLORMAP, ax=ax1)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('')
    ax1.set_title('Precision / Recall / F1 (mean across channels)')
    ax1.legend(frameon=True, title='Method')

    # Panel 2: Radar plot
    ax2 = fig.add_subplot(gs[0,1], polar=True)
    radar_metrics = METRICS
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    for _, row in radar_df.iterrows():
        values = [row[m] for m in radar_metrics]
        values = np.concatenate([values, [values[0]]])
        ax2.plot(angles, values, label=row['method'], color=COLORMAP[row['method']])
        ax2.fill(angles, values, alpha=0.15, color=COLORMAP[row['method']])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([
        'F1', 'Precision', 'Recall', 'FR', 'ISI (inv)', 'Presence', 'Amp.CV (inv)', 'SNR'
    ], fontsize=9)
    ax2.set_yticklabels([])
    ax2.set_title('Radar (normalized; inverted bad metrics)')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Panel 3: Heatmap of z-score normalized metrics
    ax3 = fig.add_subplot(gs[0,2])
    z_df = agg_mean.copy()
    z_vals = []
    for metric in METRICS:
        col = z_df[metric]
        z = (col - col.mean()) / (col.std() + 1e-9)
        if metric in INVERT_METRICS:
            z = -z  # invert to keep higher=better visual logic
        z_vals.append(z.values)
    heat = np.vstack(z_vals)
    im = ax3.imshow(heat, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2)
    ax3.set_yticks(range(len(METRICS)))
    ax3.set_yticklabels(['F1','Precision','Recall','FiringRate','ISI(inv)','Presence','AmpCV(inv)','SNR'])
    ax3.set_xticks(range(len(z_df)))
    ax3.set_xticklabels(z_df['method'].tolist(), rotation=45, ha='right')
    ax3.set_title('Metric Profile (z-score; inverted bad metrics)')
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)
    cbar.set_label('Z-score')

    # Panel 4: SNR vs Amplitude CV scatter
    ax4 = fig.add_subplot(gs[1,0])
    sns.scatterplot(data=df, x='mean_amplitude_cv', y='mean_snr', hue='method', palette=COLORMAP, ax=ax4, s=70, alpha=0.85)
    ax4.set_xlabel('Amplitude CV')
    ax4.set_ylabel('SNR')
    ax4.set_title('SNR vs Amplitude CV (per channel)')
    ax4.legend(frameon=True)

    # Panel 5: Presence ratio vs Firing rate
    ax5 = fig.add_subplot(gs[1,1])
    sns.scatterplot(data=df, x='mean_firing_rate', y='mean_presence_ratio', hue='method', palette=COLORMAP, ax=ax5, s=70, alpha=0.85)
    ax5.set_xlabel('Firing Rate (Hz)')
    ax5.set_ylabel('Presence Ratio')
    ax5.set_title('Firing Rate vs Presence Ratio')
    ax5.legend().remove()

    # Panel 6: ISI violation vs F1
    ax6 = fig.add_subplot(gs[1,2])
    sns.scatterplot(data=df, x='mean_isi_violation', y='overall_f1_score', hue='method', palette=COLORMAP, ax=ax6, s=70, alpha=0.85)
    ax6.set_xlabel('ISI Violation Rate')
    ax6.set_ylabel('F1 Score')
    ax6.set_title('F1 vs ISI Violation')
    ax6.legend().remove()

    fig.suptitle('Spike Sorting Method Comparison (Channel-averaged & Distributions)', fontsize=16, y=0.99)
    fig.tight_layout(rect=[0,0,1,0.97])

    out_file = out_dir / 'comparison_multimetrics.png'
    fig.savefig(out_file, dpi=300)
    print(f"Saved figure: {out_file}")

    # Save aggregated metrics CSV
    agg_mean.to_csv(out_dir / 'metrics_aggregated_means.csv', index=False)
    df.to_csv(out_dir / 'metrics_channel_level.csv', index=False)
    print(f"Saved metrics CSVs to {out_dir}")

def main():
    all_rows = []
    for method, base in METHOD_SPECS:
        df_method = load_method_channel_metrics(base, method)
        if df_method.empty:
            print(f"Warning: no data found for method {method} at {base}")
        all_rows.append(df_method)
    df = pd.concat(all_rows, ignore_index=True)
    # Drop rows with all NaNs in core metrics
    core = ['overall_f1_score','overall_precision','overall_recall']
    df = df.dropna(subset=core, how='all')
    out_dir = Path('output/visualizations/comparison')
    plot_comparison(df, out_dir)

if __name__ == '__main__':
    main()
