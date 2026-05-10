"""
Sweep orchestrator: iterate over m setting configs × k teacher criteria,
running all 5 pipeline stages for each combination.

Calls scripts 01-05 via direct Python imports (not subprocess) for efficiency
when run single-threaded, or dispatches to subprocess when --jobs > 1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_sweep(
    settings_dir: str | Path,
    output_dir: str | Path,
    n_channels: Optional[int] = None,
    student_model: str = "gpt-4o",
    teacher_model: str = "gpt-4o",
    provider: str = "gpt4o",
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
    adapter: str = "qwen35",
    jobs: int = 1,
    force: bool = False,
    use_mock: bool = False,
) -> dict:
    """
    Run the full pipeline over all setting YAMLs in settings_dir.

    Args:
        settings_dir: Directory containing per-setting YAML files.
        output_dir: Root output directory.
        n_channels: Override n_channels from setting YAMLs (useful for quick runs).
        student_model: Student VLM model name.
        teacher_model: Teacher VLM model name.
        provider: VLM provider.
        n_train: Override n_train_channels.
        n_eval: Override n_eval_channels.
        adapter: Adapter name for Stage 4 ("qwen35" or "gemma4").
        jobs: Number of parallel jobs (1 = sequential).
        force: Re-run all stages even if outputs exist.
        use_mock: Use mock VLM responses.

    Returns:
        Sweep results summary dict.
    """
    settings_dir = Path(settings_dir)
    output_dir = Path(output_dir)

    yaml_files = sorted(settings_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No setting YAMLs found in {settings_dir}")

    print(f"\n{'='*60}")
    print(f"SWEEP: {len(yaml_files)} settings, jobs={jobs}")
    print(f"{'='*60}\n")

    tasks = []
    for yaml_file in yaml_files:
        overrides = {}
        if n_channels is not None:
            overrides["n_channels"] = n_channels
        if n_train is not None:
            overrides["n_train"] = n_train
        if n_eval is not None:
            overrides["n_eval"] = n_eval

        tasks.append({
            "setting_yaml": str(yaml_file),
            "output_dir": str(output_dir),
            "student_model": student_model,
            "teacher_model": teacher_model,
            "provider": provider,
            "adapter": adapter,
            "force": force,
            "use_mock": use_mock,
            **overrides,
        })

    if jobs == 1:
        results = [_run_one_setting(t) for t in tasks]
    else:
        results = _run_parallel(tasks, jobs)

    from .aggregator import aggregate_sweep_results
    summary = aggregate_sweep_results(output_dir)

    sweep_file = output_dir / "sweep_summary.json"
    with open(sweep_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[sweep done] Summary written to {sweep_file}")
    return summary


def _run_one_setting(task: dict) -> dict:
    """Run all pipeline stages for one setting sequentially."""
    setting_yaml = task["setting_yaml"]
    output_dir = task["output_dir"]
    force_flag = ["--force"] if task.get("force") else []
    mock_flag = ["--use-mock"] if task.get("use_mock") else []

    stages = [
        [sys.executable, "scripts/01_simulate.py",
         "--config", setting_yaml,
         "--output-dir", output_dir,
         *force_flag],
        [sys.executable, "scripts/02_build_actions.py",
         "--config", setting_yaml,
         "--output-dir", output_dir,
         "--all-channels",
         *force_flag],
        [sys.executable, "scripts/03_run_trajectories.py",
         "--config", setting_yaml,
         "--output-dir", output_dir,
         "--all-channels",
         "--student-model", task.get("student_model", "gpt-4o"),
         "--teacher-model", task.get("teacher_model", "gpt-4o"),
         "--provider", task.get("provider", "gpt4o"),
         *mock_flag,
         *force_flag],
        [sys.executable, "scripts/04_adapt.py",
         "--config", setting_yaml,
         "--output-dir", output_dir,
         "--adapter", task.get("adapter", "qwen35"),
         *force_flag],
        [sys.executable, "scripts/05_evaluate_alignment.py",
         "--config", setting_yaml,
         "--output-dir", output_dir,
         "--all-channels",
         "--student-model", task.get("student_model", "gpt-4o"),
         "--provider", task.get("provider", "gpt4o"),
         *mock_flag,
         *force_flag],
    ]

    for cmd in stages:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [error] Stage failed: {' '.join(cmd)}")
            return {"setting_yaml": setting_yaml, "status": "failed", "failed_stage": cmd[1]}

    return {"setting_yaml": setting_yaml, "status": "ok"}


def _run_parallel(tasks: list[dict], jobs: int) -> list[dict]:
    """Run settings in parallel using a process pool."""
    from multiprocessing.pool import Pool

    with Pool(processes=jobs) as pool:
        results = pool.map(_run_one_setting, tasks)
    return results
