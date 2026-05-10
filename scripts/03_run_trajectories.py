"""
Stage 3: Run teacher-student interaction trajectories.

For each channel, the student VLM makes zero-shot decisions; the GT-aware
teacher provides calibrated feedback. Both are recorded step-by-step.

Output per channel: output/<setting_id>/<ch_id>/trajectory/trajectory.jsonl

Usage:
  uv run python scripts/03_run_trajectories.py \\
      --config configs/settings/setting_001.yaml \\
      --student-model gpt-4o --teacher-model gpt-4o \\
      --all-channels
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulate.setting import SettingConfig
from src.trajectories.runner import TrajectoryRunner
from src.trajectories.student import StudentRunner
from src.trajectories.teacher import GroundTruthTeacher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Run teacher-student interaction trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to per-setting YAML.")
    parser.add_argument("--global-config", default="config.yaml")
    parser.add_argument("--channel-id", default=None)
    parser.add_argument("--all-channels", action="store_true")
    parser.add_argument("--student-model", default="gpt-4o")
    parser.add_argument("--teacher-model", default="gpt-4o")
    parser.add_argument("--provider", default="gpt4o", help="VLM provider (gpt4o | claude | openrouter | vllm).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--auto-discard-threshold", type=int, default=500)
    parser.add_argument("--small-cluster-threshold", type=int, default=4000)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--use-mock", action="store_true", help="Mock VLM calls (no API key needed).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--enable-rag-baseline", action="store_true")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-waveform-weight", type=float, default=0.7)
    parser.add_argument("--rag-feature-weight", type=float, default=0.3)
    parser.add_argument("--rag-memory-path", default=None)
    parser.add_argument(
        "--rag-persist-memory",
        action="store_true",
        help="Persist RAG memory to JSONL. Default is in-memory only.",
    )
    parser.add_argument(
        "--rag-overwrite-memory",
        action="store_true",
        help="If persisting memory, clear memory before trajectory run.",
    )
    args = parser.parse_args()

    if not args.channel_id and not args.all_channels:
        parser.error("Specify --channel-id <id> or --all-channels.")

    cfg = SettingConfig.load(
        args.config,
        global_config=args.global_config if Path(args.global_config).exists() else None,
    )

    output_dir = Path(args.output_dir)
    setting_dir = output_dir / cfg.setting_id
    rag_memory_path = None
    if args.rag_persist_memory or args.rag_memory_path:
        rag_memory_path = (
            Path(args.rag_memory_path)
            if args.rag_memory_path
            else (setting_dir / "rag_memory.jsonl")
        )

    if args.all_channels:
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
    teacher = GroundTruthTeacher(
        cfg=cfg,
        teacher_model=args.teacher_model,
        provider=args.provider,
        use_mock=args.use_mock,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    runner = TrajectoryRunner(
        cfg=cfg,
        student=student,
        teacher=teacher,
        auto_discard_threshold=args.auto_discard_threshold,
        small_cluster_threshold=args.small_cluster_threshold,
        force=args.force,
        enable_rag_baseline=args.enable_rag_baseline,
        rag_top_k=args.rag_top_k,
        rag_waveform_weight=args.rag_waveform_weight,
        rag_feature_weight=args.rag_feature_weight,
        rag_memory_path=rag_memory_path if args.enable_rag_baseline else None,
        rag_overwrite_memory=args.rag_overwrite_memory if args.enable_rag_baseline else False,
    )

    print(f"\n{'='*60}")
    print(f"Stage 3 — Trajectories: {cfg.setting_id}")
    print(f"  channels       : {len(channel_ids)}")
    print(f"  student_model  : {args.student_model}")
    print(f"  teacher_model  : {args.teacher_model}")
    print(f"  teacher_style  : {cfg.teacher_style}")
    print(f"  use_mock       : {args.use_mock}")
    print(f"  rag_enabled    : {args.enable_rag_baseline}")
    if args.enable_rag_baseline:
        print(f"  rag_top_k      : {args.rag_top_k}")
        print(f"  rag_weights    : waveform={args.rag_waveform_weight}, feature={args.rag_feature_weight}")
        if rag_memory_path is None:
            print("  rag_memory     : in-memory only")
        else:
            print(f"  rag_memory     : {rag_memory_path}")
        print(f"  rag_overwrite  : {args.rag_overwrite_memory}")
    print(f"{'='*60}\n")

    for ch_id in channel_ids:
        raw_dir = setting_dir / ch_id / "raw"
        if not raw_dir.exists():
            print(f"  [skip] {ch_id}: raw/ not found.")
            continue
        print(f"  Running trajectory for {ch_id}...")
        runner.run(raw_dir=raw_dir, output_dir=setting_dir / ch_id / "trajectory")

    print(f"\nStage 3 complete.")


if __name__ == "__main__":
    main()
