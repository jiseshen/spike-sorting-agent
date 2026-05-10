# Script Entry Points

## New numbered entrypoints (MEArec simulation pipeline)

Run these in order for the full research pipeline:

```bash
# Stage 1: Generate MEArec recordings + overclustering
uv run python scripts/01_simulate.py --config configs/settings/setting_001.yaml

# Stage 2: Build ground-truth action trajectories
uv run python scripts/02_build_actions.py --config configs/settings/setting_001.yaml --all-channels

# Stage 3: Run teacher-student interaction
uv run python scripts/03_run_trajectories.py --config configs/settings/setting_001.yaml \
  --student-model gpt-4o --teacher-model gpt-4o --all-channels

# Stage 4: Few-shot adaptation
uv run python scripts/04_adapt.py --config configs/settings/setting_001.yaml

# Stage 5: Evaluate alignment
uv run python scripts/05_evaluate_alignment.py --config configs/settings/setting_001.yaml --all-channels

# Stage 6: Sweep over all settings
uv run python scripts/06_sweep.py --settings-dir configs/settings/ --jobs 4
```

## Legacy scripts (real MATLAB data)

### `run/`
End-to-end pipeline runs on real CH3/CH20/CH30/CH31 data:
- `run_single_channel.py` — pure VLM pipeline on one channel (configurable via CLI arg)
- `run_all_channels.py` — pure VLM on all channels (multiprocessing)
- `run_baseline_pipeline.py` — VLM + heuristics baseline on all channels
- `run_ablation_no_metrics.py` — VLM without numerical metrics
- `run_single_channel_qwen35.py` — single channel with finetuned Qwen3.5 backbone
- `run_qwen35_full_cycle.sh` — sequential: base eval → finetune → finetuned eval
- `run_gemma4_full_cycle.sh` — Gemma-4 full cycle
- `run_rag_backbone_unit_compare.sh` — vLLM replay unit test (`no_rag` vs `rag`) for Qwen3.5-4B + Gemma4-E4B, with auto plotting

```bash
uv run python scripts/run/run_single_channel.py CH3
uv run python scripts/run/run_all_channels.py
```

### `finetune/`
Dataset construction and model training:
- `build_finetune_dataset.py` — build JSONL + image assets from MATLAB action sheets
- `prepare_hf_dataset.py` — convert to HuggingFace datasets format
- `split_finetune_dataset_by_channel.py` — train/eval split by channel
- `train_qwen35_unsloth.py` — Unsloth Vision SFT for Qwen3.5-4B
- `train_gemma4_unsloth.py` — Unsloth Vision SFT for Gemma-4-E4B

### `aggregate/`
- `aggregate_results.py` — cross-channel metrics aggregation
- `aggregate_baseline.py` — aggregate baseline pipeline results

### `analysis/`
- `compute_human_curation.py` — analyze human curation patterns
- `evaluate_baseline.py` — evaluate baseline pipeline
- `generate_curation_stats_table.py` — statistics tables
- `visualize_clusters.py` — visualize cluster waveforms

### `plot/`
- `plot_ablation.py` — ablation test results
- `plot_comparison.py` — pipeline performance comparison
- `plot_f1_scores.py` — F1 score plots
- `plot_results.py` — overall results
- `plot_sft_accuracy_grouped.py` — grouped before/after accuracy bars
- `plot_rag_backbone_unit_compare.py` — RAG/no-RAG bars + overall compare + step learning curves (Qwen/Gemma)

### `test/`
- `eval_unit_actions_from_dataset.py` — unit-test style action accuracy
- `test_vlm_unit_decisions.py` — VLM decision unit tests

### `demo/`
- `demo_simulated_teacher_feedback.py` — one-shot teacher feedback demo
- `run_channel_teacher_budget.py` — limited teacher interaction budget demo
