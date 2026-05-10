"""
Stage 4: Few-shot adaptation from trajectory demonstrations.

Samples n_train channels from the setting, builds an SFT dataset from their
trajectories, converts to HuggingFace format, and optionally launches training.

Output: output/<setting_id>/adapters/<run_id>/

Usage:
  # Build dataset only (default)
  uv run python scripts/04_adapt.py --config configs/settings/setting_001.yaml

  # Build dataset + launch training
  uv run python scripts/04_adapt.py --config configs/settings/setting_001.yaml \\
      --adapter qwen35 --train

  # Eval-only (skip dataset build, use existing adapter)
  uv run python scripts/04_adapt.py --config configs/settings/setting_001.yaml \\
      --eval-only --adapter-path output/setting_001/adapters/n10_seed0_qwen35
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulate.setting import SettingConfig
from src.adapt.sampler import sample_channels, build_sft_dataset_from_split
from src.adapt.sft import prepare_hf_dataset, launch_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Few-shot adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to per-setting YAML.")
    parser.add_argument("--global-config", default="config.yaml")
    parser.add_argument(
        "--n-train", type=int, default=None,
        help="Number of train channels (overrides setting YAML).",
    )
    parser.add_argument(
        "--n-eval", type=int, default=None,
        help="Number of eval channels (overrides setting YAML).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for train/eval split (overrides setting YAML).",
    )
    parser.add_argument("--adapter", default="qwen35", choices=["qwen35", "gemma4"])
    parser.add_argument("--train", action="store_true", help="Launch SFT training after building dataset.")
    parser.add_argument("--eval-only", action="store_true", help="Skip dataset build; use existing adapter.")
    parser.add_argument("--adapter-path", default=None, help="Existing adapter path (for --eval-only).")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = SettingConfig.load(
        args.config,
        global_config=args.global_config if Path(args.global_config).exists() else None,
    )

    n_train = args.n_train or cfg.n_train_channels
    n_eval = args.n_eval or cfg.n_eval_channels
    seed = args.seed if args.seed is not None else cfg.split_seed
    run_id = f"n{n_train}_seed{seed}_{args.adapter}"

    print(f"\n{'='*60}")
    print(f"Stage 4 — Adapt: {cfg.setting_id}")
    print(f"  n_train : {n_train}")
    print(f"  n_eval  : {n_eval}")
    print(f"  seed    : {seed}")
    print(f"  adapter : {args.adapter}")
    print(f"  run_id  : {run_id}")
    print(f"{'='*60}\n")

    if args.eval_only:
        if not args.adapter_path:
            parser.error("--eval-only requires --adapter-path.")
        print(f"  Eval-only mode. Adapter at: {args.adapter_path}")
        return

    split = sample_channels(
        setting_id=cfg.setting_id,
        output_dir=args.output_dir,
        n_train=n_train,
        n_eval=n_eval,
        seed=seed,
    )

    adapter_dir = build_sft_dataset_from_split(
        split=split,
        output_dir=args.output_dir,
        adapter_run_id=run_id,
    )

    hf_dir = prepare_hf_dataset(adapter_dir=adapter_dir)

    if args.train:
        rc = launch_training(adapter_dir=adapter_dir, adapter_name=args.adapter)
        if rc != 0:
            print(f"  [error] Training exited with code {rc}")
            sys.exit(rc)
    else:
        print(f"  Dataset ready at {hf_dir}. Pass --train to launch SFT training.")

    print(f"\nStage 4 complete. Adapter run: {adapter_dir}")


if __name__ == "__main__":
    main()
