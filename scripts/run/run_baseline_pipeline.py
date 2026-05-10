"""
Run Baseline VLM Pipeline on all channels (CH3, CH20, CH30, CH31).

Baseline approach:
- VLM only judges neuronal validity (KEEP/DISCARD)
- Split decisions use heuristic (ISI violation > 0.5%)
- Merge decisions use heuristic (correlation > 0.3, ISI < 0.5%)

Automatically processes each channel in parallel and saves results to separate folders.
Cross-channel aggregation removed from this script; use `uv run python scripts/aggregate/aggregate_results.py` when needed.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from multiprocessing import Pool

from src.io.matlab_loader import convert_mat_to_sortings
from src.cluster.manager import ClusterManager
from src.pipeline.baseline import BaselineVLMCurationPipeline

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Please set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

load_dotenv(override=True)
print(os.getenv("DOTENV_LOAD_VALID"))

# =====================================================================
# CONFIGURATION
# =====================================================================

USE_MOCK = False  # Set to True for testing without API calls
PROVIDER = "gpt4o"  # "gpt4o" or "claude"
MODEL = "gpt-5.1"  # "gpt-4o", "gpt-4o-mini", etc.

# Channels to process
CHANNELS = ["CH3", "CH20", "CH30", "CH31"]

# Output base directory
OUTPUT_BASE = Path("output/baseline_vlm_all_channels")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# =====================================================================
# MAIN PROCESSING
# =====================================================================

def process_single_channel(channel):
    """
    Process a single channel. This function will be called in parallel.
    
    Returns:
        Dict with stats, or None if failed
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {channel}")
    print(f"{'='*80}")
    
    # Start timer
    start_time = time.time()
    
    # Setup paths
    data_file = f"data/{channel}_spikes.mat"
    channel_output = OUTPUT_BASE / channel
    channel_output.mkdir(exist_ok=True)
    
    try:
        # [1] Load data
        print(f"\n[1] Loading {data_file}...")
        data_path = Path(data_file)
        if not data_path.exists():
            print(f"  ✗ File not found: {data_file}")
            return None
        
        try:
            sorting, sorting_tree, meta = convert_mat_to_sortings(str(data_path))
        except Exception as e:
            print(f"  ✗ Failed to load {data_file}: {e}")
            print(f"  Skipping {channel}...")
            return None
        
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
        
        # [3] Initialize baseline pipeline
        print(f"\n[3] Initializing baseline pipeline...")
        pipeline = BaselineVLMCurationPipeline(
            manager=manager,
            waveforms=waveforms,
            spike_times=spike_times_all,
            sampling_rate=Fs,
            provider=PROVIDER,
            model=MODEL,
            use_mock=USE_MOCK,
        )
        print(f"  ✓ Baseline pipeline initialized")
        
        # [4] Run pipeline
        print(f"\n[4] Running baseline pipeline...")
        if USE_MOCK:
            print(f"  📝 Using MOCK VLM responses (no API calls)")
        else:
            print(f"  🌐 Using REAL VLM API: {PROVIDER}/{MODEL}")
            print(f"  ⚠ This will make API calls for neuronal validity checks!")
        
        pipeline_start = time.time()
        manager = pipeline.run(output_dir=channel_output)
        final_clusters = manager.get_active_clusters()
        pipeline_time = time.time() - pipeline_start
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print(f"\n  ✓ Pipeline complete: {len(final_clusters)} final clusters")
        print(f"  ⏱ Pipeline execution time: {pipeline_time:.1f}s ({pipeline_time/60:.1f}min)")
        print(f"  ⏱ Total time (including I/O): {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
        
        # [5] Save results
        print(f"\n[5] Saving results to {channel_output}...")
        
        # Assignments and hierarchy
        np.save(channel_output / "final_assigns.npy", manager.assigns)
        np.save(channel_output / "final_hierarchy_tree.npy", manager.hierarchy_tree)
        np.save(channel_output / "overcluster_assigns.npy", manager.overcluster_assigns)
        
        print(f"  ✓ Results saved")
        print(f"  ⏱ Skipping evaluation (run scripts/aggregate/aggregate_results.py later)")
        
        # Record channel stats
        stats = manager.get_statistics()
        stats_dict = {
            'channel': channel,
            'n_initial_clusters': len(initial_clusters),
            'n_final_clusters': len(final_clusters),
            'n_spikes_total': len(spike_times_all),
            'n_spikes_assigned': stats['n_spikes_assigned'],
            'n_spikes_noise': stats['n_spikes_noise'],
            'n_operations': stats['n_operations'],
            'elapsed_time_s': elapsed_time,
        }
        
        print(f"\n✓ {channel} complete!")
        
        return {
            'stats': stats_dict,
            'success': True,
        }
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ {channel} interrupted by user")
        return None
    
    except Exception as e:
        print(f"\n\n✗ {channel} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("="*80)
    print("BASELINE VLM PIPELINE - BATCH PROCESSING (PARALLEL)")
    print("="*80)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"Use mock: {USE_MOCK}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Parallel workers: {len(CHANNELS)} (one per channel)")
    print("="*80)

    # Collect lightweight per-channel stats (for quick overview only)
    channel_stats = []

    # Process channels in parallel
    print(f"\n🚀 Starting parallel processing of {len(CHANNELS)} channels...")
    start_total = time.time()

    with Pool(processes=len(CHANNELS)) as pool:
        results = pool.map(process_single_channel, CHANNELS)

    total_time = time.time() - start_total
    print(f"\n⏱ Total parallel processing time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Collect results
    for result in results:
        if result is not None and result['success']:
            channel_stats.append(result['stats'])

    print("\n" + "-"*80)
    print("AGGREGATION SKIPPED")
    print("-"*80)
    print("Run `uv run python scripts/aggregate/aggregate_results.py` to compute cross-channel summaries when desired.")
    if channel_stats:
        print("\nChannel summary (counts only):")
        for s in channel_stats:
            print(f"  {s['channel']}: initial={s['n_initial_clusters']} final={s['n_final_clusters']} time={s['elapsed_time_s']:.1f}s")

    print("\n" + "="*80)
    print("BASELINE PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_BASE.absolute()}")
    print("Next: aggregate with `uv run python scripts/aggregate/aggregate_results.py` if needed.")
    print("="*80)

