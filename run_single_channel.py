"""
Run Pure VLM Pipeline on a single channel.

Usage: uv run run_single_channel.py CH3
       uv run run_single_channel.py CH20
       uv run run_single_channel.py CH30
       uv run run_single_channel.py CH31
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from src.matlab_loader import convert_mat_to_sortings
from src.cluster_manager import ClusterManager
from src.cluster_features import ClusterFeatures
from src.vlm_curation_pipeline_pure import PureVLMCurationPipeline
from src.metrics import generate_full_evaluation_report, print_evaluation_summary
import spikeinterface as si

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Please set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# =====================================================================
# CONFIGURATION
# =====================================================================

USE_MOCK = False  # Set to True for testing without API calls
PROVIDER = "gpt4o"  # "gpt4o" or "claude"
MODEL = "gpt-4.1"  # "gpt-4o", "gpt-4o-mini", "o1-preview", etc.
TEMPERATURE = 0.0  # For vision models
REASONING_EFFORT = None  # For reasoning models: "low"/"medium"/"high"

load_dotenv(override=True)

# Get channel from command line
if len(sys.argv) < 2:
    print("ERROR: Please specify a channel")
    print("Usage: uv run run_single_channel.py CH3")
    print("       uv run run_single_channel.py CH20")
    print("       uv run run_single_channel.py CH30")
    print("       uv run run_single_channel.py CH31")
    sys.exit(1)

CHANNEL = sys.argv[1]

# Validate channel
VALID_CHANNELS = ["CH3", "CH20", "CH30", "CH31"]
if CHANNEL not in VALID_CHANNELS:
    print(f"ERROR: Invalid channel '{CHANNEL}'")
    print(f"Valid channels: {', '.join(VALID_CHANNELS)}")
    sys.exit(1)

# Output base directory
OUTPUT_BASE = Path(f"output/main_{MODEL}")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# =====================================================================
# MAIN PROCESSING
# =====================================================================

print("="*80)
print(f"PURE VLM PIPELINE - SINGLE CHANNEL: {CHANNEL}")
print("="*80)
print(f"Provider: {PROVIDER}")
print(f"Model: {MODEL}")
print(f"Use mock: {USE_MOCK}")
print(f"Output directory: {OUTPUT_BASE}")
print("="*80)

# Start timer
start_time = time.time()

# Setup paths
data_file = f"data/{CHANNEL}_spikes.mat"
channel_output = OUTPUT_BASE / CHANNEL
channel_output.mkdir(exist_ok=True)

try:
    # [1] Load data
    print(f"\n[1] Loading {data_file}...")
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"  ✗ File not found: {data_file}")
        sys.exit(1)
    
    sorting, sorting_tree, meta = convert_mat_to_sortings(str(data_path))
    waveforms = meta["waveforms"]
    spike_times_all = meta["spiketimes"]
    Fs = meta["Fs"]
    hierarchy_tree = meta["hierarchy_tree"]
    gt_assigns = meta.get("curation_assigns")
    
    print(f"  ✓ Loaded {len(spike_times_all)} spikes")
    print(f"  ✓ Sampling rate: {Fs} Hz")
    if gt_assigns is not None:
        gt_cluster_ids = np.unique(gt_assigns[gt_assigns > 0])
        print(f"  ✓ Ground truth: {len(gt_cluster_ids)} clusters")
    
    # [2] Initialize manager
    print(f"\n[2] Initializing cluster manager...")
    manager = ClusterManager(
        initial_assigns=meta['hierarchy_assigns'],
        overcluster_assigns=meta['overcluster_assigns'],
        hierarchy_tree=hierarchy_tree,
        spike_times=spike_times_all,
        waveforms=waveforms,
    )
    initial_clusters = manager.get_active_clusters()
    print(f"  ✓ Initial clusters: {len(initial_clusters)}")
    
    # [3] Initialize features
    print(f"\n[3] Computing cluster features...")
    features = ClusterFeatures(
        meta=meta,
        assigns=manager.assigns,
    )
    print(f"  ✓ Features ready")
    
    # [4] Initialize pipeline
    print(f"\n[4] Initializing pipeline...")
    pipeline = PureVLMCurationPipeline(
        manager=manager,
        features=features,
        sampling_rate=Fs,
        auto_discard_threshold=500,
        small_cluster_threshold=4000,
        final_minimum_threshold=5000,
        provider=PROVIDER,
        model=MODEL,
        use_mock=USE_MOCK,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        output_dir=channel_output,
    )
    print(f"  ✓ Pipeline initialized")
    
    # [5] Run pipeline
    print(f"\n[5] Running full pipeline...")
    if USE_MOCK:
        print(f"  📝 Using MOCK VLM responses (no API calls)")
    else:
        print(f"  🌐 Using REAL VLM API: {PROVIDER}/{MODEL}")
        print(f"  ⚠ This will make many API calls and may take 5-15 minutes!")
    
    pipeline_start = time.time()
    final_clusters = pipeline.run_full_pipeline()
    pipeline_time = time.time() - pipeline_start
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"\n  ✓ Pipeline complete: {len(final_clusters)} final clusters")
    print(f"  ⏱ Pipeline execution time: {pipeline_time:.1f}s ({pipeline_time/60:.1f}min)")
    print(f"  ⏱ Total time (including I/O): {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    
    # [6] Save results
    print(f"\n[6] Saving results to {channel_output}...")
    
    # Action log
    pipeline.save_action_log(channel_output / "action_log.csv")
    
    # Assignments and hierarchy
    np.save(channel_output / "final_assigns.npy", manager.assigns)
    np.save(channel_output / "final_hierarchy_tree.npy", manager.hierarchy_tree)
    np.save(channel_output / "overcluster_assigns.npy", manager.overcluster_assigns)
    
    print(f"  ✓ Results saved")
    
    # [7] Evaluation
    print(f"\n[7] Running evaluation...")
    
    if final_clusters:
        # Create final sorting
        spike_frames = (spike_times_all * Fs).astype(np.int64)
        final_sorting = si.NumpySorting.from_unit_dict(
            {int(cid): spike_frames[manager.assigns == cid] for cid in final_clusters},
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
            assigns=manager.assigns,
            ground_truth_sorting=gt_sorting,
            gt_assigns=gt_assigns,
            sampling_frequency=Fs,
            output_dir=channel_output,
        )
        
        print_evaluation_summary(report)
    else:
        print(f"  ⚠ No final clusters - skipping evaluation")
    
    # Record channel stats
    stats = manager.get_statistics()
    print(f"\n{'='*80}")
    print(f"CHANNEL STATISTICS")
    print(f"{'='*80}")
    print(f"Initial clusters: {len(initial_clusters)}")
    print(f"Final clusters: {len(final_clusters)}")
    print(f"Total spikes: {len(spike_times_all)}")
    print(f"Assigned spikes: {stats['n_spikes_assigned']}")
    print(f"Noise spikes: {stats['n_spikes_noise']}")
    print(f"Operations: {stats['n_operations']}")
    print(f"Elapsed time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    print(f"{'='*80}")
    
    print(f"\n✓ {CHANNEL} processing complete!")
    print(f"Results saved to: {channel_output.absolute()}")
    
except KeyboardInterrupt:
    print(f"\n\n⚠ {CHANNEL} interrupted by user")
    sys.exit(1)

except Exception as e:
    print(f"\n\n✗ {CHANNEL} failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
