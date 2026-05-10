"""
Stage 2: Build ground-truth action trajectories.

For each channel, repeatedly queries the oracle to find the optimal
split/merge/discard sequence that recovers the MEArec ground-truth labels.

Output per channel: output/<setting_id>/<ch_id>/actions/actions.jsonl

Usage:
  uv run python scripts/02_build_actions.py --config configs/settings/setting_001.yaml --all-channels
  uv run python scripts/02_build_actions.py --config configs/settings/setting_001.yaml --channel-id ch_000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulate.setting import SettingConfig
from src.actions.trajectory import build_gt_trajectory
from src.actions.validator import validate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Build GT action trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to per-setting YAML.")
    parser.add_argument("--global-config", default="config.yaml")
    parser.add_argument(
        "--channel-id", default=None,
        help="Single channel ID to process (e.g. ch_000). Mutually exclusive with --all-channels.",
    )
    parser.add_argument("--all-channels", action="store_true", help="Process all channels.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--purity-threshold", type=float, default=0.80,
        help="Cluster purity threshold for SPLIT vs KEEP decision.",
    )
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
        channel_ids = sorted(
            d.name for d in setting_dir.iterdir()
            if d.is_dir() and d.name.startswith("ch_")
        )
    else:
        channel_ids = [args.channel_id]

    print(f"\n{'='*60}")
    print(f"Stage 2 — Build Actions: {cfg.setting_id}")
    print(f"  channels     : {len(channel_ids)}")
    print(f"  purity_thresh: {args.purity_threshold}")
    print(f"{'='*60}\n")

    for ch_id in channel_ids:
        raw_dir = setting_dir / ch_id / "raw"
        if not raw_dir.exists():
            print(f"  [skip] {ch_id}: raw/ not found. Run Stage 1 first.")
            continue

        print(f"  Building trajectory for {ch_id}...")
        steps = build_gt_trajectory(
            raw_dir=raw_dir,
            purity_threshold=args.purity_threshold,
            force=args.force,
        )

        validation = validate_trajectory(steps)
        if not validation.valid:
            print(f"  [WARN] {ch_id} trajectory has errors:")
            print(str(validation))

    print(f"\nStage 2 complete.")


if __name__ == "__main__":
    main()
