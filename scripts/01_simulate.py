"""
Stage 1: Simulate extracellular recordings with MEArec.

For each channel index (0 … n_channels-1):
  1. Generate a MEArec recording with the setting's noise/drift/overlap parameters
  2. Run spike sorting (MountainSort5) to produce overclustering
  3. Save numpy arrays compatible with ClusterManager to output/<setting_id>/<ch_id>/raw/

Usage:
  uv run python scripts/01_simulate.py --config configs/settings/setting_001.yaml
  uv run python scripts/01_simulate.py --config configs/settings/setting_001.yaml \\
      --n-channels 5 --output-dir output/ --force
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulate.setting import SettingConfig
from src.simulate.generator import generate_recording
from src.simulate.overcluster import overcluster_recording


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate MEArec recordings + overclustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to per-setting YAML (e.g. configs/settings/setting_001.yaml).",
    )
    parser.add_argument(
        "--global-config", default="config.yaml",
        help="Path to global config.yaml (merged under per-setting YAML).",
    )
    parser.add_argument(
        "--n-channels", type=int, default=None,
        help="Override n_channels from the setting YAML.",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Root output directory.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate channels even if outputs already exist.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.n_channels is not None:
        overrides = {"simulation": {"n_channels": args.n_channels}}

    cfg = SettingConfig.load(
        setting_yaml=args.config,
        global_config=args.global_config if Path(args.global_config).exists() else None,
        overrides=overrides if overrides else None,
    )

    import yaml
    setting_out = Path(args.output_dir) / cfg.setting_id
    setting_out.mkdir(parents=True, exist_ok=True)
    with open(setting_out / "setting_config.yaml", "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Stage 1 — Simulate: {cfg.setting_id}")
    print(f"  n_channels   : {cfg.n_channels}")
    print(f"  noise_level  : {cfg.noise_level} uV RMS ({cfg.noise_label})")
    print(f"  drift        : {'enabled' if cfg.drift_enabled else 'disabled'}")
    print(f"  probe        : {cfg.probe}")
    print(
        f"  hierarchy    : {'enabled' if cfg.hierarchy_enabled else 'disabled'} "
        f"(metric={cfg.hierarchy_similarity_metric}, min_sim={cfg.hierarchy_min_similarity})"
    )
    print(f"  output_dir   : {args.output_dir}")
    print(f"{'='*60}\n")

    for ch_idx in range(cfg.n_channels):
        print(f"[{ch_idx+1}/{cfg.n_channels}] Generating ch_{ch_idx:03d}...")
        generate_recording(cfg, ch_idx, args.output_dir, force=args.force)

        print(f"[{ch_idx+1}/{cfg.n_channels}] Overclustering ch_{ch_idx:03d}...")
        overcluster_recording(cfg, ch_idx, args.output_dir, force=args.force)

    print(f"\nStage 1 complete. Outputs in {Path(args.output_dir) / cfg.setting_id}/")


if __name__ == "__main__":
    main()
