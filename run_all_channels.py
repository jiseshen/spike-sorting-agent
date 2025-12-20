"""
Run Pure VLM Pipeline on all channels (CH3, CH20, CH30, CH31).

Automatically processes each channel and saves results to separate folders.
Cross-channel aggregation removed from this script; use `uv run aggregate_results.py` when needed.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count

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
MODEL = "gpt-5.1"  # "gpt-4o", "gpt-4o-mini", "o1-preview", etc.
TEMPERATURE = 0.0  # For vision models
REASONING_EFFORT = "medium"  # For reasoning models: "low"/"medium"/"high"

load_dotenv(override=True)
print(os.getenv("DOTENV_LOAD_VALID"))

# Channels to process
CHANNELS = ["CH31"]

# Output base directory
OUTPUT_BASE = Path(f"output/main_{MODEL}")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# =====================================================================
# MAIN PROCESSING
# =====================================================================

def process_single_channel(channel):
    """
    Process a single channel. This function will be called in parallel.
    
    Returns:
        Dict with results and stats, or None if failed
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
        
        result_dict = None
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
            
            # Store for aggregation
            if report['overall_performance'] is not None:
                perf = report['overall_performance']
                perf['channel'] = channel
                perf['n_initial_clusters'] = len(initial_clusters)
                perf['n_final_clusters'] = len(final_clusters)
                perf['elapsed_time_s'] = elapsed_time
                result_dict = perf
        else:
            print(f"  ⚠ No final clusters - skipping evaluation")
        
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
            'result': result_dict,
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
    print("PURE VLM PIPELINE - BATCH PROCESSING (PARALLEL)")
    print("="*80)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"Use mock: {USE_MOCK}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Parallel workers: {len(CHANNELS)} (one per channel)")
    print("="*80)

    # Collect lightweight per-channel stats (for quick overview only)
    all_results = []
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
            if result['result'] is not None:
                all_results.append(result['result'])
            channel_stats.append(result['stats'])

    print("\n" + "-"*80)
    print("AGGREGATION SKIPPED")
    print("-"*80)
    print("Run `uv run aggregate_results.py` to compute cross-channel summaries when desired.")
    if channel_stats:
        print("\nChannel summary (counts only):")
        for s in channel_stats:
            print(f"  {s['channel']}: initial={s['n_initial_clusters']} final={s['n_final_clusters']} time={s['elapsed_time_s']:.1f}s")

    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_BASE.absolute()}")
    print("Next: aggregate with `uv run aggregate_results.py` if needed.")
    print("="*80)
