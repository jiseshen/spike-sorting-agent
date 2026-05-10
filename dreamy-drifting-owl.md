# Project Reorganization Plan: SpikeSorting

## Context

The SpikeSorting project is transitioning from a system that curates real MATLAB recordings using VLMs to a full research pipeline supporting six stages: simulated data generation (MEArec), canonical action space construction, teacher-student interaction trajectories, few-shot adaptation, step-level alignment evaluation, and heterogeneous-lab scaling. The reorganization makes each stage a first-class module with a numbered entrypoint, while preserving all working code.

---

## Directory Structure

### `src/` — Library code

Existing modules stay **unchanged** (or lightly extended):

```
src/
  io/
    matlab_loader.py          # unchanged
    mearec_loader.py          # NEW: load MEArec .h5 + GT templates
  cluster/                    # unchanged
  agent/
    api.py, runner.py, context.py  # unchanged
    teacher_feedback.py            # extended by Stage 3
  eval/
    metrics.py                # extended by Stage 5
  pipeline/                   # unchanged
  ablation/                   # unchanged
```

Six new stage modules added under `src/`:

```
src/
  simulate/           # Stage 1
    generator.py      # MEArec wrapper: generate recordings per setting
    overcluster.py    # Run MountainSort5 → overcluster_assigns + hierarchy_tree
    setting.py        # SettingConfig dataclass; merges global config + per-setting YAML

  actions/            # Stage 2
    trajectory.py     # Build GT action trajectory from gt_assigns + ClusterManager state
    oracle.py         # Stateless: given current state + GT → best next action
    validator.py      # Check trajectory consistency

  trajectories/       # Stage 3
    runner.py         # TrajectoryRunner: steps student + teacher per channel
    teacher.py        # GroundTruthTeacher: wraps teacher_feedback.py + GT reasoning
    student.py        # StudentRunner: thin wrapper over agent/runner.py VLM calls
    record.py         # TrajectoryStep dataclass; serialize to .jsonl

  adapt/              # Stage 4
    sampler.py        # Sample n channels from a setting for train/eval split
    sft.py            # Build HF dataset + call finetune scripts
    evaluator.py      # Replay student on held-out channels; collect predicted actions

  alignment/          # Stage 5
    action_metrics.py    # Step-level accuracy, confusion matrix, edit distance to GT
    reasoning_metrics.py # Cosine sim / LLM-as-judge between student and teacher rationale
    report.py            # Aggregate per-channel alignment_report.json

  scale/              # Stage 6
    sweep.py          # Outer loop over m settings × k teacher_criteria
    aggregator.py     # Cross-setting aggregation → sweep_summary.json
```

### `scripts/` — Entrypoints (top-level, not inside `src/`)

New numbered entrypoints for each stage:

```
scripts/
  01_simulate.py           # Generate MEArec recordings + overclustering
  02_build_actions.py      # Build GT action trajectories
  03_run_trajectories.py   # Run teacher-student interaction loops
  04_adapt.py              # SFT adaptation on sampled trajectories
  05_evaluate_alignment.py # Step-level action accuracy + reasoning alignment
  06_sweep.py              # Orchestrate all stages over m settings
```

Existing scripts move from `src/scripts/` to top-level `scripts/` as-is:

```
scripts/
  run/         # run_single_channel.py, run_all_channels.py, etc.
  finetune/    # build_finetune_dataset.py, train_gemma4_unsloth.py, etc.
  aggregate/   # aggregate_results.py, aggregate_baseline.py
  analysis/    # compute_human_curation.py, evaluate_baseline.py, etc.
  plot/        # plot_ablation.py, plot_comparison.py, etc.
  demo/        # demo_simulated_teacher_feedback.py (superseded by Stage 3 but kept)
  test/        # eval_unit_actions_from_dataset.py, test_vlm_unit_decisions.py
```

---

## Entrypoint CLI Specs

| Script | Key args | Writes to |
|---|---|---|
| `01_simulate.py` | `--setting-id`, `--config`, `--n-channels`, `--seed`, `--output-dir` | `output/<setting_id>/<ch_id>/raw/` |
| `02_build_actions.py` | `--setting-id`, `--channel-id` / `--all-channels`, `--output-dir` | `output/<setting_id>/<ch_id>/actions/actions.jsonl` |
| `03_run_trajectories.py` | `--setting-id`, `--channel-id`, `--student-model`, `--teacher-model`, `--provider` | `output/<setting_id>/<ch_id>/trajectory/trajectory.jsonl` |
| `04_adapt.py` | `--setting-id`, `--n-train`, `--n-eval`, `--seed`, `--adapter`, `--eval-only` | `output/<setting_id>/adapters/<run_id>/` |
| `05_evaluate_alignment.py` | `--setting-id`, `--channel-id` / `--all-channels`, `--model`, `--adapter-path` | `output/<setting_id>/<ch_id>/eval/alignment_report.json` |
| `06_sweep.py` | `--settings-dir`, `--n-channels`, `--teacher-criteria`, `--jobs` | `output/sweep_summary.json` |

---

## Output Layout

```
output/
  <setting_id>/
    setting_config.yaml
    <channel_id>/
      raw/
        recording.h5, metadata.json
        overcluster_assigns.npy, hierarchy_assigns.npy, hierarchy_tree.npy
        gt_assigns.npy, waveforms.npy, spike_times.npy
      actions/
        actions.jsonl           # {"step", "action_type", "cluster_id", "gt_reasoning"}
        trajectory_summary.json
      trajectory/
        trajectory.jsonl        # TrajectoryStep per line
        trajectory_summary.json
      eval/
        alignment_report.json   # {"action_accuracy", "reasoning_cosine_sim", "llm_judge_score"}
        action_confusion_matrix.csv
        quality_metrics.csv
    adapters/
      <n10_seed0_qwen35>/
        model_checkpoint/
        train_dataset.jsonl, eval_dataset.jsonl
  sweep_summary.json
```

---

## Stage Interface Boundaries

- **Stage 1 → Stage 2**: numpy arrays (overcluster_assigns, gt_assigns, waveforms, spike_times) — exact schema already consumed by `ClusterManager.__init__()`
- **Stage 2 → Stage 3**: `actions.jsonl` lines with `{step, action_type, cluster_id, gt_reasoning}`
- **Stage 3 → Stage 4**: `trajectory.jsonl` lines with `TrajectoryStep` (student_action, student_rationale, teacher_feedback, gt_action, image_paths)
- **Stage 4 → Stage 5**: HF model checkpoint path + eval channel list
- **Stage 5 → Stage 6**: `alignment_report.json` per channel

---

## Configuration Schema

Global defaults remain in `config.yaml`. Per-setting YAMLs live in `configs/settings/<setting_id>.yaml` and override only changed fields:

```yaml
# configs/settings/setting_001.yaml
setting_id: setting_001
simulation:
  n_channels: 20
  duration: 120
  n_neurons_range: [5, 20]
  probe: "Neuronexus-32"
  seed: 42
noise:
  level: "low"
  noise_level: 10.0  # uV RMS
drift:
  enabled: false
  drift_velocity: 0.0
overlap:
  enabled: true
  max_overlap_pairs: 3
artifact:
  type: "none"  # none | periodic | burst
teacher_criteria:
  min_spike_count: 500
  max_isi_violation_rate: 0.006
  require_biphasic: true
  min_snr: 3.0
  teacher_style: "strict"  # strict | liberal | snr_only
adaptation:
  n_train_channels: 10
  n_eval_channels: 5
  split_seed: 0
```

`src/simulate/setting.py` merges global config + per-setting YAML + CLI overrides, in that priority order.

---

## Key Files to Reuse

- [src/cluster/manager.py](src/cluster/manager.py) — shared state for all stages; no changes needed
- [src/agent/runner.py](src/agent/runner.py) — `vlm_phase1_cluster_decision` / `vlm_phase2_merge_decision` called directly from `src/trajectories/student.py`
- [src/agent/teacher_feedback.py](src/agent/teacher_feedback.py) — imported by `src/trajectories/teacher.py`; add GT-aware reasoning reference field
- [src/eval/metrics.py](src/eval/metrics.py) — `match_clusters_to_ground_truth` imported by `src/alignment/action_metrics.py`
- [src/scripts/demo/demo_simulated_teacher_feedback.py](src/scripts/demo/demo_simulated_teacher_feedback.py) — the proof-of-concept `TrajectoryRunner` loop to generalize

---

## Verification

1. `python scripts/01_simulate.py --setting-id setting_001 --config configs/settings/setting_001.yaml --n-channels 2 --output-dir output/` → verify `output/setting_001/ch_000/raw/*.npy` exist
2. `python scripts/02_build_actions.py --setting-id setting_001 --all-channels` → verify `actions.jsonl` has valid step entries
3. `python scripts/03_run_trajectories.py --setting-id setting_001 --channel-id ch_000 --student-model gpt-4o` → verify `trajectory.jsonl` lines contain both student and teacher fields
4. `python scripts/05_evaluate_alignment.py --setting-id setting_001 --all-channels` → verify `alignment_report.json` contains `action_accuracy` and `reasoning_cosine_sim`
5. `python scripts/06_sweep.py --settings-dir configs/settings/ --jobs 1` → verify `sweep_summary.json` contains per-setting aggregated rows
