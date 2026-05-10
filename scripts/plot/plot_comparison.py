"""
Method Comparison Figure: Hierarchical vs Baseline-VLM vs GPT-5.1

Methods:
- baseline (before_curation): Hierarchical clustering only
- baseline-VLM (baseline_vlm_all_channels): Neural validation + heuristic
- gpt-5.1 (main_gpt-5.1): VLM agent (reasoning model with metrics)

Metrics (8 total):
- overall_f1_score, overall_precision, overall_recall
- mean_firing_rate, mean_isi_violations, mean_presence_ratio
- mean_amplitude_cv, mean_snr

Figure: 1x3 grid
- Left: Bar chart (Precision/Recall/F1)
- Middle: Radar plot with visible scale ticks
- Right: Quality metrics comparison
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Nice style
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science','no-latex','grid'])
except Exception:
    plt.style.use('seaborn-v0_8-whitegrid')

METHOD_SPECS = [
    ("Hierarchical", "output/before_curation"),
    ("NeuHeu", "output/baseline_vlm_all_channels"),
    ("AMPS-GPT5.1", "output/main_gpt-5.1"),
    ("Human", "output/human_curation"),
]

METRICS = [
    "overall_f1_score",
    "overall_precision",
    "overall_recall",
    "mean_isi_violations",
    "mean_presence_ratio",
    "mean_amplitude_cv",
    "mean_snr",
]

# Metrics where lower is better → invert for radar
INVERT_METRICS = {"mean_isi_violations", "mean_amplitude_cv"}

COLORMAP = {
    "Hierarchical": "#888888",
    "NeuHeu": "#ff7f0e",
    "AMPS-GPT5.1": "#d62728",
    "Human": "#2ca02c",
}

def parse_aggregate_performance(file_path: Path) -> dict:
    """Parse aggregate_performance.txt file (mean ± std format)."""
    if not file_path.exists():
        return {}

    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('=') and not line.startswith('-'):
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value_str = parts[1].strip()

            # Parse "0.1234 ± 0.0456" format
            match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', value_str)
            if match:
                mean_val = float(match.group(1))
                std_val = float(match.group(2))
                data[key] = {'mean': mean_val, 'std': std_val}
            else:
                # Try simple float (some lines like Counts may not match)
                try:
                    data[key] = {'mean': float(value_str), 'std': 0.0}
                except ValueError:
                    pass

    return data

def load_method_aggregates(method_name: str, base_dir: str):
    """Load aggregated means/stds from aggregate_performance.txt for a method.

    Returns:
        means_row: dict with keys 'method' + METRICS
        stds_row: dict with keys 'method' + METRICS
    """
    file_path = Path(base_dir) / "aggregate_performance.txt"
    parsed = parse_aggregate_performance(file_path)
    if not parsed:
        return None, None

    means_row = {'method': method_name}
    stds_row = {'method': method_name}
    for m in METRICS:
        if m in parsed:
            means_row[m] = parsed[m]['mean']
            stds_row[m] = parsed[m]['std']
        else:
            means_row[m] = np.nan
            stds_row[m] = np.nan

    return means_row, stds_row

def load_method_data(method_name: str, base_dir: str) -> pd.DataFrame:
    """Load per-channel data from aggregate_summary.csv."""
    csv_path = Path(base_dir) / "aggregate_summary.csv"
    df = load_aggregate_summary_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    
    # Add method column
    df['method'] = method_name
    
    # Ensure all required metrics exist
    required_cols = ['channel'] + METRICS
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    return df[['method', 'channel'] + METRICS]

def prepare_radar_values(agg_mean: pd.DataFrame) -> pd.DataFrame:
    """Prepare radar values using Human as baseline (ratio calculation).
    
    - For precision/recall/F1: use raw values (already 0-1 scale)
    - For quality metrics: calculate ratio to Human baseline
      - Normal metrics (firing_rate, presence, SNR): value/human
      - Inverted metrics (ISI, CV): human/value (so lower is better becomes higher ratio)
    """
    # Find Human baseline
    human_row = agg_mean[agg_mean['method'] == 'Human']
    if human_row.empty:
        print("⚠ Warning: Human baseline not found, using min-max normalization")
        # Fallback to old logic
        norm_rows = []
        for _, row in agg_mean.iterrows():
            norm_row = {'method': row['method']}
            for metric in METRICS:
                vals = agg_mean[metric].values
                min_v, max_v = np.nanmin(vals), np.nanmax(vals)
                if max_v - min_v < 1e-9:
                    norm = 0.5
                else:
                    norm = (row[metric] - min_v) / (max_v - min_v + 1e-9)
                norm_row[metric] = norm
            norm_rows.append(norm_row)
        return pd.DataFrame(norm_rows)
    
    gt_metrics = ['overall_precision', 'overall_recall', 'overall_f1_score']
    quality_metrics = ['mean_isi_violations', 'mean_presence_ratio', 'mean_amplitude_cv', 'mean_snr']
    
    norm_rows = []
    for _, row in agg_mean.iterrows():
        norm_row = {'method': row['method']}
        
        # GT metrics: use raw values (already 0-1)
        for metric in gt_metrics:
            norm_row[metric] = row[metric]
        
        # Quality metrics: ratio to Human
        for metric in quality_metrics:
            human_val = human_row[metric].values[0]
            method_val = row[metric]
            
            if human_val < 1e-9:  # Avoid division by zero
                norm_row[metric] = 0.5
            elif metric in INVERT_METRICS:
                # Lower is better → human/method (so lower method value = higher ratio)
                norm_row[metric] = human_val / (method_val + 1e-9)
            else:
                # Higher is better → method/human
                norm_row[metric] = method_val / human_val
        
        norm_rows.append(norm_row)
    
    return pd.DataFrame(norm_rows)

def plot_comparison(agg_mean: pd.DataFrame, agg_std: pd.DataFrame, out_dir: Path):
    """Generate comparison figure with 3 panels using aggregated means/stds."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare radar data
    radar_df = prepare_radar_values(agg_mean)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    # Panel 1: Bar chart (Precision/Recall/F1) - exclude Human (always 1.0)
    ax1 = fig.add_subplot(gs[0,0])
    pr_metrics = ['overall_precision','overall_recall','overall_f1_score']
    x_pos = np.arange(len(pr_metrics))
    width = 0.25
    methods = agg_mean['method'].tolist()
    methods_for_bar = [m for m in methods if m != 'Human']

    for i, method in enumerate(methods_for_bar):
        means = [agg_mean[agg_mean['method'] == method][m].values[0] for m in pr_metrics]
        stds = [agg_std[agg_std['method'] == method][m].values[0] for m in pr_metrics]
        ax1.bar(x_pos + i*width, means, width, yerr=stds, label=method, 
                color=COLORMAP[method], alpha=0.8, capsize=3)
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_xlabel('')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(['Precision','Recall','F1'], fontsize=10)
    ax1.set_title('(A) Ground Truth Performance', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fontsize=9, loc='upper left')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Radar plot with visible ticks
    ax2 = fig.add_subplot(gs[0,1], polar=True)
    angles = np.linspace(0, 2*np.pi, len(METRICS), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    
    radar_labels = [
        'F1', 'Precision', 'Recall',
        'ISI↓', 'Presence', 'Amp.CV↓', 'SNR'
    ]
    
    for _, row in radar_df.iterrows():
        values = [row[m] for m in METRICS]
        values_closed = values + [values[0]]
        method = row['method']
        ax2.plot(angles_closed, values_closed, 'o-', label=method, 
                color=COLORMAP[method], linewidth=2.5, markersize=5)
        ax2.fill(angles_closed, values_closed, alpha=0.15, color=COLORMAP[method])
    
    ax2.set_xticks(angles)
    ax2.set_xticklabels(radar_labels, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=9, color='gray')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title('(B) Multi-Metric Profile\n(normalized; ↓ = inverted)', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), frameon=True, fontsize=10)
    
    fig.suptitle('Method Comparison', 
                fontsize=15, fontweight='bold', y=1.02)
    
    out_file = out_dir / 'comparison_methods.png'
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_file}")
    
    # Save data
    agg_mean.to_csv(out_dir / 'comparison_methods_means.csv', index=False)
    agg_std.to_csv(out_dir / 'comparison_methods_stds.csv', index=False)
    print(f"✓ Saved CSVs to {out_dir}")

def main():
    mean_rows = []
    std_rows = []

    for method, base in METHOD_SPECS:
        means_row, stds_row = load_method_aggregates(method, base)
        if means_row is None:
            print(f"⚠ Warning: no aggregate_performance for {method} at {base}")
            continue
        mean_rows.append(means_row)
        std_rows.append(stds_row)

    if not mean_rows:
        print("Error: No valid aggregated data loaded")
        return

    agg_mean = pd.DataFrame(mean_rows)
    agg_std = pd.DataFrame(stds_row for stds_row in std_rows)

    print("\nAggregated means:")
    print(agg_mean[['method'] + METRICS])

    out_dir = Path('output/visualizations/comparison')
    plot_comparison(agg_mean, agg_std, out_dir)

if __name__ == '__main__':
    main()
