"""
Ablation Study Figure: GPT-5.1 (with metrics) vs GPT-5.1 (no metrics) vs GPT-4.1

Methods:
- GPT-5.1 (main_gpt-5.1): Full agent with quality metrics in prompts
- Ablation (ablation_no_metrics): GPT-5.1 without quality metrics in prompts
- GPT-4.1 (main_gpt-4.1): Non-reasoning model with quality metrics

Primary Focus: Ground truth comparison metrics (precision, recall, F1)
Secondary: Radar plot showing all metrics (normalized)

Figure: 1x2 grid
- Left: Bar chart (Precision/Recall/F1) - raw values with error bars
- Right: Radar plot with visible scale ticks (normalized)
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
    ("AMPS-GPT5.1", "output/main_gpt-5.1"),
    ("NoMetrics", "output/ablation_no_metrics"),
    ("NonReasoning", "output/main_gpt-4.1"),
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
    "AMPS-GPT5.1": "#d62728",
    "NoMetrics": "#9467bd",
    "NonReasoning": "#1f77b4",
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
                # Try simple float
                try:
                    data[key] = {'mean': float(value_str), 'std': 0.0}
                except ValueError:
                    pass

    return data

def load_method_aggregates(method_name: str, base_dir: str):
    """Load aggregated means/stds from aggregate_performance.txt for a method."""
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

def prepare_radar_values(agg_mean: pd.DataFrame) -> pd.DataFrame:
    """Normalize metrics to [0,1] for radar plot; invert bad metrics."""
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
                    # Lower is better → invert
                    norm = 1.0 - (raw - min_v) / (max_v - min_v)
                else:
                    norm = (raw - min_v) / (max_v - min_v)
            norm_row[metric] = norm
        norm_rows.append(norm_row)
    return pd.DataFrame(norm_rows)

def plot_ablation(agg_mean: pd.DataFrame, agg_std: pd.DataFrame, out_dir: Path):
    """Generate ablation figure with GT bar chart only."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    
    # Panel 1: Bar chart (Precision/Recall/F1) - raw values
    ax1 = fig.add_subplot(111)
    pr_metrics = ['overall_precision','overall_recall','overall_f1_score']
    x_pos = np.arange(len(pr_metrics))
    width = 0.25
    methods = agg_mean['method'].tolist()

    for i, method in enumerate(methods):
        means = [agg_mean[agg_mean['method'] == method][m].values[0] for m in pr_metrics]
        stds = [agg_std[agg_std['method'] == method][m].values[0] for m in pr_metrics]
        ax1.bar(x_pos + i*width, means, width, yerr=stds, label=method, 
                color=COLORMAP[method], alpha=0.8, capsize=3)
    
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(['Precision','Recall','F1 Score'], fontsize=11)
    ax1.set_title('Ground Truth Performance', fontsize=13, fontweight='bold')
    ax1.legend(frameon=True, fontsize=10, loc='upper left')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Ablation Study: Model & Prompt Design', 
                fontsize=15, fontweight='bold', y=0.98)
    
    out_file = out_dir / 'ablation_study.png'
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_file}")
    
    # Save data
    agg_mean.to_csv(out_dir / 'ablation_study_means.csv', index=False)
    agg_std.to_csv(out_dir / 'ablation_study_stds.csv', index=False)
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

    out_dir = Path('output/visualizations/ablation')
    plot_ablation(agg_mean, agg_std, out_dir)

if __name__ == '__main__':
    main()
