"""
Evaluate baseline (pre-curation) hierarchy clustering results.
Uses hierarchy.assigns directly from MATLAB without any VLM curation.
"""

from pathlib import Path
import numpy as np
from src.matlab_loader import convert_mat_to_sortings
from src.metrics import generate_full_evaluation_report
import spikeinterface as si

# Configuration
CHANNELS = ["CH3", "CH20", "CH30", "CH31"]
OUTPUT_DIR = Path("output/before_curation")

def evaluate_hierarchy_baseline():
    """Evaluate hierarchy clustering (before any curation)."""
    
    print("="*80)
    print("EVALUATING BASELINE HIERARCHY CLUSTERING (PRE-CURATION)")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    for channel in CHANNELS:
        print(f"\n{'='*80}")
        print(f"Processing {channel}")
        print(f"{'='*80}")
        
        # Load data
        data_file = Path(f"data/{channel}_spikes.mat")
        if not data_file.exists():
            print(f"  ✗ Data file not found: {data_file}")
            continue
        
        try:
            # Load sortings and metadata
            hierarchy_sorting, gt_sorting, meta = convert_mat_to_sortings(str(data_file))
            
            waveforms = meta["waveforms"]
            spike_times = meta["spiketimes"]
            Fs = meta["Fs"]
            hierarchy_assigns = meta["hierarchy_assigns"]
            gt_assigns = meta.get("curation_assigns")
            
            if gt_assigns is None:
                print(f"  ⚠ No ground truth available for {channel}")
                continue
            
            # Get cluster info
            hierarchy_ids = hierarchy_sorting.get_unit_ids()
            gt_ids = gt_sorting.get_unit_ids()
            
            print(f"  Hierarchy clusters: {len(hierarchy_ids)}")
            print(f"  Ground truth clusters: {len(gt_ids)}")
            
            # Create output directory for this channel
            channel_output = OUTPUT_DIR / channel
            channel_output.mkdir(parents=True, exist_ok=True)
            
            # Save hierarchy assigns for reference
            np.save(channel_output / "hierarchy_assigns.npy", hierarchy_assigns)
            
            # Generate evaluation report
            report = generate_full_evaluation_report(
                curated_sorting=hierarchy_sorting,
                waveforms=waveforms,
                spike_times=spike_times,
                assigns=hierarchy_assigns,
                ground_truth_sorting=gt_sorting,
                gt_assigns=gt_assigns,
                sampling_frequency=Fs,
                output_dir=channel_output,
            )
            
            # Print summary
            if report['overall_performance'] is not None:
                perf = report['overall_performance']
                print(f"\n  Performance:")
                print(f"    Precision: {perf['overall_precision']:.4f}")
                print(f"    Recall:    {perf['overall_recall']:.4f}")
                print(f"    F1 Score:  {perf['overall_f1_score']:.4f}")
                print(f"    Clusters:  {perf['n_curated_clusters']} hierarchy -> {perf['n_matched_clusters']} matched to GT")
            
            # Print quality metrics summary
            if report['quality_metrics'] is not None:
                qm = report['quality_metrics']
                print(f"\n  Quality Metrics:")
                print(f"    Mean firing rate:    {qm['firing_rate'].mean():.2f} Hz")
                print(f"    Mean ISI violations: {qm['isi_violations_rate'].mean():.2%}")
                print(f"    Mean SNR:            {qm['snr'].mean():.2f}")
            
            print(f"\n  ✓ Results saved to {channel_output}")
            
        except Exception as e:
            print(f"  ✗ Error processing {channel}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("BASELINE EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    evaluate_hierarchy_baseline()
