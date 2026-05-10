"""
Stage 5: Evaluate step-level action prediction and reasoning alignment.

Replays the student model on held-out channels, then measures:
  - Per-step action accuracy vs ground truth
  - Confusion matrix across {SPLIT, MERGE, DISCARD, KEEP}
  - Sequence-level edit distance
  - Reasoning cosine similarity (or LLM-as-judge) between student
    rationale and teacher feedback

Output per channel: output/<setting_id>/<ch_id>/eval/alignment_report.json

Usage:
  uv run python scripts/05_evaluate_alignment.py \\
      --config configs/settings/setting_001.yaml \\
      --all-channels --student-model gpt-4o

  # With finetuned adapter (vLLM/OpenRouter):
  uv run python scripts/05_evaluate_alignment.py \\
      --config configs/settings/setting_001.yaml \\
      --all-channels --student-model <finetuned-model-id> --provider openrouter
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulate.setting import SettingConfig
from src.trajectories.student import StudentRunner
from src.adapt.evaluator import evaluate_on_channels
from src.alignment.report import generate_alignment_report, aggregate_setting_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Evaluate action accuracy + reasoning alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to per-setting YAML.")
    parser.add_argument("--global-config", default="config.yaml")
    parser.add_argument("--channel-id", default=None)
    parser.add_argument("--all-channels", action="store_true")
    parser.add_argument("--student-model", default="gpt-4o")
    parser.add_argument("--provider", default="gpt4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--adapter-path", default=None,
        help="Optional HF adapter directory (for finetuned model evaluation).",
    )
    parser.add_argument(
        "--reasoning-method", default="cosine", choices=["cosine", "llm_judge"],
        help="Method for measuring reasoning alignment.",
    )
    parser.add_argument(
        "--llm-judge-model", default=None,
        help="Model for LLM-as-judge scoring (only used with --reasoning-method llm_judge).",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.channel_id and not args.all_channels:
        parser.error("Specify --channel-id <id> or --all-channels.")

    cfg = SettingConfig.load(
        args.config,
        global_config=args.global_config if Path(args.global_config).exists() else None,
    )

    output_dir = Path(args.output_dir)
    setting_dir = output_dir / cfg.setting_id

    if args.all_channels:
        # Use eval channels from adapter run if available; else use all channels
        adapter_dirs = sorted((setting_dir / "adapters").glob("*/eval_dataset.jsonl")) if (setting_dir / "adapters").exists() else []
        if adapter_dirs:
            with open(adapter_dirs[-1]) as f:
                eval_meta = json.load(f)
            channel_ids = eval_meta.get("eval_channel_ids", [])
            print(f"  Using eval channels from adapter: {adapter_dirs[-1].parent.name}")
        else:
            channel_ids = sorted(
                d.name for d in setting_dir.iterdir()
                if d.is_dir() and d.name.startswith("ch_")
                and (d / "raw" / "waveforms.npy").exists()
            )
    else:
        channel_ids = [args.channel_id]

    student = StudentRunner(
        student_model=args.student_model,
        provider=args.provider,
        use_mock=args.use_mock,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )

    print(f"\n{'='*60}")
    print(f"Stage 5 — Evaluate Alignment: {cfg.setting_id}")
    print(f"  channels         : {len(channel_ids)}")
    print(f"  student_model    : {args.student_model}")
    print(f"  reasoning_method : {args.reasoning_method}")
    print(f"  use_mock         : {args.use_mock}")
    print(f"{'='*60}\n")

    # Step 1: Run student predictions on held-out channels
    evaluate_on_channels(
        setting_id=cfg.setting_id,
        channel_ids=channel_ids,
        output_dir=args.output_dir,
        student=student,
        force=args.force,
    )

    # Step 2: Compute alignment metrics
    for ch_id in channel_ids:
        pred_file = setting_dir / ch_id / "eval" / "eval_predictions.jsonl"
        if not pred_file.exists():
            print(f"  [skip] {ch_id}: no predictions found.")
            continue

        generate_alignment_report(
            setting_id=cfg.setting_id,
            channel_id=ch_id,
            output_dir=args.output_dir,
            reasoning_method=args.reasoning_method,
            llm_judge_model=args.llm_judge_model,
            provider=args.provider,
            use_mock=args.use_mock,
            force=args.force,
        )

    # Step 3: Aggregate across channels
    summary = aggregate_setting_reports(
        setting_id=cfg.setting_id,
        output_dir=args.output_dir,
    )
    print(
        f"\n  Setting summary: "
        f"action_acc={summary.get('mean_action_accuracy', 0):.3f}, "
        f"reasoning_sim={summary.get('mean_reasoning_sim', 0):.3f}"
    )
    print(f"\nStage 5 complete.")


if __name__ == "__main__":
    main()
