"""
Generate a table with curation statistics for each channel in GPT-5.1 output.

Columns:
- Channel
- Total Spikes
- Over-clusters (initial)
- Human Curation Actions (from spikes/curation/actions)
- VLM Calls (from action_log.csv, excluding initial Phase0 DISCARDs)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from src.matlab_loader import load_matlab_spikes

CHANNELS = ["CH3", "CH20", "CH30", "CH31"]
GPT51_BASE = Path("output/main_gpt-5.1")
HUMAN_BASE = Path("output/human_curation")

def count_human_curation_actions(channel: str) -> int:
    """Count number of human curation actions from spikes/curation/actions or action."""
    mat_file = Path(f"data/{channel}_spikes.mat")
    if not mat_file.exists():
        return 0
    
    # Try HDF5 format first
    try:
        with h5py.File(mat_file, 'r') as f:
            # Try both 'actions' (plural) and 'action' (singular)
            for key in ['spikes/curation/actions', 'spikes/curation/action']:
                if key in f:
                    actions = f[key][:]
                    # actions is (1, n_actions) array of object references
                    return actions.shape[1] if actions.ndim == 2 else len(actions)
    except:
        # Try v5 MAT file format
        try:
            from scipy.io import loadmat
            md = loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
            spikes = md.get('spikes')
            if spikes is not None:
                curation = getattr(spikes, 'curation', None)
                if curation is not None:
                    # Try both 'actions' (plural) and 'action' (singular)
                    for attr_name in ['actions', 'action']:
                        actions = getattr(curation, attr_name, None)
                        if actions is not None:
                            actions_arr = np.array(actions)
                            return len(actions_arr) if actions_arr.ndim == 1 else actions_arr.shape[1]
        except Exception as e:
            print(f"Warning: Could not read curation actions for {channel}: {e}")
            return 0
    
    return 0

def count_vlm_calls(channel: str) -> int:
    """Count VLM calls from action_log.csv, excluding initial Phase0 DISCARDs."""
    action_log = GPT51_BASE / channel / "action_log.csv"
    if not action_log.exists():
        return 0
    
    df = pd.read_csv(action_log)
    
    # Find the first non-DISCARD action or first non-Phase0 action
    # Strategy: skip all initial Phase0 DISCARDs, then count everything else
    first_non_phase0_idx = None
    for idx, row in df.iterrows():
        if row['Phase'] != 'Phase0':
            first_non_phase0_idx = idx
            break
    
    if first_non_phase0_idx is None:
        # All actions are Phase0 (shouldn't happen in real data)
        return 0
    
    # Count from first non-Phase0 action onward
    vlm_actions = df.iloc[first_non_phase0_idx:]
    return len(vlm_actions)

def get_spike_count(channel: str) -> int:
    """Get total spike count from MATLAB file."""
    mat_file = Path(f"data/{channel}_spikes.mat")
    if not mat_file.exists():
        return 0
    
    data = load_matlab_spikes(str(mat_file))
    spiketimes = data.get('spiketimes')
    
    if spiketimes is None:
        return 0
    
    return len(spiketimes)

def get_overcluster_count(channel: str) -> int:
    """Get initial over-cluster count from overcluster_assigns."""
    mat_file = Path(f"data/{channel}_spikes.mat")
    if not mat_file.exists():
        return 0
    
    data = load_matlab_spikes(str(mat_file))
    overcluster_assigns = data.get('overcluster_assigns')
    
    if overcluster_assigns is None:
        return 0
    
    # Count unique non-noise clusters (cluster_id > 0)
    unique_clusters = np.unique(overcluster_assigns[overcluster_assigns > 0])
    return len(unique_clusters)

def main():
    rows = []
    
    for channel in CHANNELS:
        spike_count = get_spike_count(channel)
        overcluster_count = get_overcluster_count(channel)
        human_actions = count_human_curation_actions(channel)
        vlm_calls = count_vlm_calls(channel)
        
        rows.append({
            'Channel': channel,
            'Total Spikes': spike_count,
            'Over-clusters': overcluster_count,
            'Human Actions': human_actions,
            'VLM Calls': vlm_calls,
        })
        
        print(f"{channel}: {spike_count} spikes, {overcluster_count} over-clusters, "
              f"{human_actions} human actions, {vlm_calls} VLM calls")
    
    df = pd.DataFrame(rows)
    
    # Add summary row
    summary = {
        'Channel': 'Total',
        'Total Spikes': df['Total Spikes'].sum(),
        'Over-clusters': df['Over-clusters'].sum(),
        'Human Actions': df['Human Actions'].sum(),
        'VLM Calls': df['VLM Calls'].sum(),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    
    # Save to CSV
    out_dir = Path("output/visualizations/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "curation_statistics.csv"
    df.to_csv(out_file, index=False)
    print(f"\n✓ Saved table to {out_file}")
    
    # Also create a nicely formatted markdown table
    md_lines = ["# Curation Statistics by Channel (GPT-5.1)", ""]
    md_lines.append("| Channel | Total Spikes | Over-clusters | Human Actions | VLM Calls |")
    md_lines.append("|---------|--------------|---------------|---------------|-----------|")
    
    for _, row in df.iterrows():
        if row['Channel'] == 'Total':
            md_lines.append(f"| **{row['Channel']}** | **{row['Total Spikes']:,}** | **{row['Over-clusters']}** | **{row['Human Actions']}** | **{row['VLM Calls']}** |")
        else:
            md_lines.append(f"| {row['Channel']} | {row['Total Spikes']:,} | {row['Over-clusters']} | {row['Human Actions']} | {row['VLM Calls']} |")
    
    md_file = out_dir / "curation_statistics.md"
    with open(md_file, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"✓ Saved markdown table to {md_file}")
    
    # Print the table to console
    print("\n" + df.to_string(index=False))

if __name__ == '__main__':
    main()
