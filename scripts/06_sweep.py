"""
Stage 6: Scale to heterogeneous lab styles.

Iterates over all setting YAMLs in a configs/ directory, running Stages 1-5
for each combination of setting × teacher criteria.

Usage:
  uv run python scripts/06_sweep.py --settings-dir configs/settings/

  # Parallel execution across 4 settings:
  uv run python scripts/06_sweep.py --settings-dir configs/settings/ --jobs 4

  # Quick smoke test (2 channels, mock VLM):
  uv run python scripts/06_sweep.py --settings-dir configs/settings/ \\
      --n-channels 2 --use-mock
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scale.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: Sweep over all settings + teacher criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings-dir", required=True,
        help="Directory containing per-setting YAML files.",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--n-channels", type=int, default=None,
        help="Override n_channels for all settings (useful for quick tests).",
    )
    parser.add_argument("--student-model", default="gpt-4o")
    parser.add_argument("--teacher-model", default="gpt-4o")
    parser.add_argument("--provider", default="gpt4o")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--adapter", default="qwen35", choices=["qwen35", "gemma4"])
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs.")
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = run_sweep(
        settings_dir=args.settings_dir,
        output_dir=args.output_dir,
        n_channels=args.n_channels,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        provider=args.provider,
        n_train=args.n_train,
        n_eval=args.n_eval,
        adapter=args.adapter,
        jobs=args.jobs,
        force=args.force,
        use_mock=args.use_mock,
    )

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  settings         : {summary.get('n_settings', 0)}")
    print(f"  mean_action_acc  : {summary.get('overall', {}).get('mean_action_accuracy', 0):.3f}")
    print(f"  mean_reasoning   : {summary.get('overall', {}).get('mean_reasoning_sim', 0):.3f}")
    print(f"  results          : {args.output_dir}/sweep_summary.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
