"""
Visualize clustering results: Baseline, Ground Truth, and GPT-5.1 separately.

Creates three separate figures (one per method) showing waveform overlays
for each cluster in a clean grid layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.io.matlab_loader import convert_mat_to_sortings

def plot_cluster_waveforms(ax, waveforms, spike_times, cluster_id, sampling_rate=30000.0, max_waveforms=500):
    """Plot waveform overlay for a single cluster."""
    n_spikes, n_samples = waveforms.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000
    
    # Subsample for plotting
    if n_spikes > max_waveforms:
        indices = np.random.choice(n_spikes, max_waveforms, replace=False)
        plot_wf = waveforms[indices]
    else:
        plot_wf = waveforms
    
    # Plot individual waveforms
    for wf in plot_wf:
        ax.plot(time_ms, wf, 'steelblue', alpha=0.2, linewidth=0.5)
    
    # Plot median
    median_wf = np.median(waveforms, axis=0)
    ax.plot(time_ms, median_wf, 'darkblue', linewidth=2)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel('Amplitude (μV)', fontsize=8)
    ax.set_title(f'Cluster {cluster_id}\n({n_spikes} spikes)', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def create_grid_figure(cluster_ids, waveforms_all, spike_times_all, assigns, Fs, title, output_path):
    """Create a single figure with all clusters in a grid layout."""
    n_clusters = len(cluster_ids)
    
    # Calculate grid dimensions (aim for roughly square)
    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Flatten axes array for easier iteration
    if n_clusters == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Plot each cluster
    for i, cid in enumerate(cluster_ids):
        mask = assigns == cid
        wf = waveforms_all[mask]
        st = spike_times_all[mask]
        
        plot_cluster_waveforms(axes[i], wf, st, cid, Fs)
    
    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def visualize_channel_clusters(channel):
    """
    Create three separate visualizations for one channel:
    1. Baseline (hierarchy) clusters
    2. Ground truth (human) clusters
    3. GPT-5.1 (VLM) clusters
    """
    print(f"\n{'='*80}")
    print(f"Visualizing {channel}")
    print(f"{'='*80}")
    
    # Load data
    data_file = Path(f"data/{channel}_spikes.mat")
    hierarchy_sorting, gt_sorting, meta = convert_mat_to_sortings(str(data_file))
    
    waveforms_all = meta["waveforms"]
    spike_times_all = meta["spiketimes"]
    Fs = meta["Fs"]
    
    hierarchy_assigns = meta["hierarchy_assigns"]
    gt_assigns = meta["curation_assigns"]
    gpt51_assigns = np.load(f"output/main_gpt-5.1/{channel}/final_assigns.npy")
    
    # Get cluster IDs (exclude noise=0)
    hierarchy_ids = sorted([cid for cid in np.unique(hierarchy_assigns) if cid > 0])
    gt_ids = sorted([cid for cid in np.unique(gt_assigns) if cid > 0])
    gpt51_ids = sorted([cid for cid in np.unique(gpt51_assigns) if cid > 0])
    
    print(f"  Hierarchy clusters: {len(hierarchy_ids)}")
    print(f"  Ground truth clusters: {len(gt_ids)}")
    print(f"  GPT-5.1 clusters: {len(gpt51_ids)}")
    
    # Create three separate figures
    output_dir = Path(f"output/visualizations/{channel}")
    
    # 1. Baseline (Hierarchy)
    create_grid_figure(
        hierarchy_ids, waveforms_all, spike_times_all, hierarchy_assigns, Fs,
        f'{channel} - Baseline (Hierarchy Clustering) - {len(hierarchy_ids)} clusters',
        output_dir / "baseline.png"
    )
    
    # 2. Ground Truth (Human)
    create_grid_figure(
        gt_ids, waveforms_all, spike_times_all, gt_assigns, Fs,
        f'{channel} - Ground Truth (Human Curation) - {len(gt_ids)} clusters',
        output_dir / "ground_truth.png"
    )
    
    # 3. GPT-5.1 (VLM)
    create_grid_figure(
        gpt51_ids, waveforms_all, spike_times_all, gpt51_assigns, Fs,
        f'{channel} - GPT-5.1 (VLM Curation) - {len(gpt51_ids)} clusters',
        output_dir / "gpt5.1.png"
    )


def main():
    """Generate visualizations for all channels."""
    channels = ["CH3", "CH20", "CH30", "CH31"]
    
    print("="*80)
    print("CLUSTER VISUALIZATION: BASELINE / GROUND TRUTH / GPT-5.1")
    print("="*80)
    
    for channel in channels:
        try:
            visualize_channel_clusters(channel)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("Results saved to: output/visualizations/[channel]/")
    print("  - baseline.png (hierarchy clustering)")
    print("  - ground_truth.png (human curation)")
    print("  - gpt5.1.png (VLM curation)")
    print("="*80)


if __name__ == "__main__":
    main()
