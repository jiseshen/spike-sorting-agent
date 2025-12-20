"""
Run Pure VLM Pipeline with GPT-4o API.

Tests the complete experimental pipeline:
- Phase 0: Auto size filter (<500 spikes)
- Phase 1: VLM iterative split/discard (with waveform, ISI, aggregation tree)
- Phase 2: VLM merge decisions (with waveform comparison and merged ISI)
- Phase 3: Final size filter (≥5000 spikes)

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run_pure_vlm_pipeline.py
"""

import os
import sys
from pathlib import Path
import numpy as np

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Please set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

from src.matlab_loader import convert_mat_to_sortings
from src.cluster_manager import ClusterManager
from src.cluster_features import ClusterFeatures
from src.vlm_curation_pipeline_pure import PureVLMCurationPipeline
from src.metrics import generate_full_evaluation_report, print_evaluation_summary
import spikeinterface as si

# Configuration
USE_MOCK = False  # Set to True for testing without API calls
PROVIDER = "gpt4o"  # "gpt4o" or "claude"

# Model selection
# Vision models (with temperature control):
#   - "gpt-4o": Best quality
#   - "gpt-4o-mini": Faster, cheaper
#   - "gpt-4.1-mini": Latest mini version
# Reasoning models (with effort control, no temperature):
#   - "o1-preview": Best reasoning (expensive)
#   - "o1-mini": Cheaper reasoning
#   - "o3-mini": Latest reasoning model
MODEL = "gpt-4o"

# For vision models: temperature control
TEMPERATURE = 0.0  # 0.0 = deterministic, higher = more creative

# For reasoning models: effort control (ignored for vision models)
REASONING_EFFORT = None  # None (auto), "low", "medium", "high"

DATA_FILE = "data/CH3_spikes.mat"  # Start with CH3

print("="*80)
print("PURE VLM CURATION PIPELINE - GPT-4o")
print("="*80)
print(f"Provider: {PROVIDER}")
print(f"Model: {MODEL}")
print(f"Use mock: {USE_MOCK}")
print(f"Data file: {DATA_FILE}")
print("="*80)

# Load data
print("\n[1] Loading data...")
data_path = Path(DATA_FILE)
if not data_path.exists():
    print(f"ERROR: Data file not found: {DATA_FILE}")
    sys.exit(1)

sorting, sorting_tree, meta = convert_mat_to_sortings(str(data_path))
waveforms = meta["waveforms"]
spike_times_all = meta["spiketimes"]
Fs = meta["Fs"]
hierarchy_tree = meta["hierarchy_tree"]
gt_assigns = meta.get("curation_assigns")  # Ground truth (may be None)

print(f"✓ Loaded {len(spike_times_all)} spikes, {waveforms.shape[1]} samples per waveform")
print(f"✓ Sampling rate: {Fs} Hz")
print(f"✓ Hierarchy tree: {hierarchy_tree.shape[1]} merges")
if gt_assigns is not None:
    gt_cluster_ids = np.unique(gt_assigns[gt_assigns > 0])
    print(f"✓ Ground truth available: {len(gt_cluster_ids)} curated clusters")
else:
    print("⚠ No ground truth available")

# Initialize cluster manager
print("\n[2] Initializing cluster manager...")
manager = ClusterManager(
    initial_assigns=meta['hierarchy_assigns'],
    overcluster_assigns=meta['overcluster_assigns'],
    hierarchy_tree=hierarchy_tree,
    spike_times=spike_times_all,
    waveforms=waveforms,
)

initial_clusters = manager.get_active_clusters()
print(f"✓ Initial clusters: {len(initial_clusters)}")

# Initialize features (for future use)
print("\n[3] Computing cluster features...")
features = ClusterFeatures(
    meta=meta,
    assigns=manager.assigns,
)
print(f"✓ Features computed for {len(features.cluster_ids)} clusters")

# Initialize pipeline
print("\n[4] Initializing Pure VLM pipeline...")
if "o1" in MODEL or "o3" in MODEL:
    print(f"    Using reasoning model with effort: {REASONING_EFFORT if REASONING_EFFORT else 'auto'}")
else:
    print(f"    Using vision model with temperature: {TEMPERATURE}")

pipeline = PureVLMCurationPipeline(
    manager=manager,
    features=features,
    sampling_rate=Fs,
    auto_discard_threshold=500,      # Phase 0: discard <500
    small_cluster_threshold=4000,    # Phase 2: small <4000, large ≥4000
    final_minimum_threshold=5000,    # Phase 3: final target ≥5000
    provider=PROVIDER,
    model=MODEL,
    use_mock=USE_MOCK,
    temperature=TEMPERATURE,
    reasoning_effort=REASONING_EFFORT,
)
print("✓ Pipeline initialized")

# Run pipeline
print("\n[5] Running pipeline...")
print("\nNOTE: This will make many API calls and may take a while!")
print("      Press Ctrl+C to interrupt if needed.\n")

try:
    final_clusters = pipeline.run_full_pipeline()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"Final clusters: {len(final_clusters)}")
    
    # Print statistics
    stats = manager.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total clusters: {stats['n_clusters']}")
    print(f"  Assigned spikes: {stats['n_spikes_assigned']}")
    print(f"  Noise spikes: {stats['n_spikes_noise']}")
    print(f"  Operations performed: {stats['n_operations']}")
    
    print(f"\nCluster sizes:")
    for cid in sorted(final_clusters):
        n_spikes = stats['cluster_sizes'].get(cid, 0)
        print(f"  Cluster {cid}: {n_spikes} spikes")
    
    # Save results
    output_dir = Path("output/pure_vlm_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save action log with reason
    pipeline.save_action_log(output_dir / "action_log.csv")
    
    # 2. Save final assignments
    np.save(output_dir / "final_assigns.npy", manager.assigns)
    print(f"✓ Final assignments saved to {output_dir / 'final_assigns.npy'}")
    
    # 3. Save final hierarchy tree
    np.save(output_dir / "final_hierarchy_tree.npy", manager.hierarchy_tree)
    print(f"✓ Final hierarchy tree saved to {output_dir / 'final_hierarchy_tree.npy'}")
    
    # 4. Save overcluster assignments (for reconstruction)
    np.save(output_dir / "overcluster_assigns.npy", manager.overcluster_assigns)
    print(f"✓ Overcluster assignments saved to {output_dir / 'overcluster_assigns.npy'}")
    
    # 5. Create final sorting object for evaluation
    print("\n[6] Running evaluation...")
    
    # Create SpikeInterface sorting from final assignments
    spike_frames = (spike_times_all * Fs).astype(np.int64)
    final_sorting = si.NumpySorting.from_unit_dict(
        {int(cid): spike_frames[manager.assigns == cid] for cid in final_clusters},
        sampling_frequency=Fs
    )
    
    # Create ground truth sorting if available
    gt_sorting = None
    if gt_assigns is not None:
        gt_cluster_ids = np.unique(gt_assigns[gt_assigns > 0])
        gt_sorting = si.NumpySorting.from_unit_dict(
            {int(cid): spike_frames[gt_assigns == cid] for cid in gt_cluster_ids},
            sampling_frequency=Fs
        )
        print(f"✓ Ground truth sorting created: {len(gt_cluster_ids)} clusters")
    
    # Run evaluation (quality metrics only, no ground truth for now)
    try:
        report = generate_full_evaluation_report(
            curated_sorting=final_sorting,
            waveforms=waveforms,
            spike_times=spike_times_all,
            assigns=manager.assigns,
            ground_truth_sorting=gt_sorting,
            gt_assigns=gt_assigns,
            sampling_frequency=Fs,
            output_dir=output_dir,
        )
        
        print_evaluation_summary(report)
        print(f"\n✓ Evaluation complete - results saved to {output_dir}")
    
    except Exception as e:
        print(f"\n⚠ Evaluation failed: {e}")
        print("  (Results still saved, continuing...)")
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"All results saved to: {output_dir.absolute()}")

except KeyboardInterrupt:
    print("\n\nPipeline interrupted by user")
    print(f"Completed {len(pipeline.actions)} actions before interruption")
    sys.exit(1)

except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
