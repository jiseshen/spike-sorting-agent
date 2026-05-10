"""
Run pure VLM pipeline on one channel with Qwen3.5-4B as backbone.

Default mode:
- provider: vllm
- model: Qwen/Qwen3.5-VL-4B-Instruct
- no-thinking: enabled via VLM_EXTRA_BODY_JSON

Examples:
  uv run python scripts/run/run_single_channel_qwen35.py --channel CH3
  uv run python scripts/run/run_single_channel_qwen35.py --channel CH31 --provider openrouter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import spikeinterface as si
from dotenv import load_dotenv

from src.cluster.features import ClusterFeatures
from src.cluster.manager import ClusterManager
from src.eval.metrics import generate_full_evaluation_report, print_evaluation_summary
from src.io.matlab_loader import convert_mat_to_sortings
from src.pipeline.pure import PureVLMCurationPipeline


VALID_CHANNELS = {"CH3", "CH20", "CH30", "CH31"}
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.5-VL-4B-Instruct"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.5-vl-4b-instruct"


def _ensure_provider_env(provider: str) -> None:
    if provider == "gpt4o" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for provider=gpt4o")
    if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is required for provider=openrouter")


def _ensure_no_thinking_extra_body(provider: str, disable_thinking: bool) -> None:
    if not disable_thinking:
        return
    if provider not in {"vllm", "openrouter"}:
        return
    if os.getenv("VLM_EXTRA_BODY_JSON", "").strip():
        return

    payload = {"chat_template_kwargs": {"enable_thinking": False}}
    os.environ["VLM_EXTRA_BODY_JSON"] = json.dumps(payload, ensure_ascii=False)
    print(f"[Config] Set VLM_EXTRA_BODY_JSON={os.environ['VLM_EXTRA_BODY_JSON']}")


def _default_model_for_provider(provider: str) -> str:
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    if provider == "vllm":
        return DEFAULT_VLLM_MODEL
    return "gpt-4.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one channel with Qwen3.5 backbone.")
    parser.add_argument("--channel", default="CH3", help="Channel: CH3/CH20/CH30/CH31")
    parser.add_argument("--provider", default="vllm", choices=["gpt4o", "openrouter", "vllm", "claude"])
    parser.add_argument("--model", default="", help="Optional model override")
    parser.add_argument("--use-mock", action="store_true", help="Use mock VLM responses")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--disable-thinking", action="store_true", help="Force no-thinking mode")
    parser.add_argument("--enable-thinking", action="store_true", help="Override and keep model thinking enabled")
    parser.add_argument("--output-tag", default="qwen35_4b_no_thinking")
    parser.add_argument("--auto-discard-threshold", type=int, default=500)
    parser.add_argument("--small-cluster-threshold", type=int, default=4000)
    parser.add_argument("--final-minimum-threshold", type=int, default=5000)
    args = parser.parse_args()

    load_dotenv(override=True)

    channel = args.channel.upper()
    if channel not in VALID_CHANNELS:
        raise ValueError(f"Invalid --channel={args.channel}. Valid: {sorted(VALID_CHANNELS)}")

    if not args.use_mock:
        _ensure_provider_env(args.provider)
    disable_thinking = args.disable_thinking or not args.enable_thinking
    _ensure_no_thinking_extra_body(args.provider, disable_thinking=disable_thinking)

    model = args.model or _default_model_for_provider(args.provider)
    output_base = Path(f"output/{args.output_tag}_{args.provider}_{model.replace('/', '_')}")
    output_base.mkdir(parents=True, exist_ok=True)
    channel_output = output_base / channel
    channel_output.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"QWEN3.5 SINGLE-CHANNEL RUN: {channel}")
    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Use mock: {args.use_mock}")
    print(f"Output: {channel_output}")
    print("=" * 80)

    start_time = time.time()
    data_file = Path(f"data/{channel}_spikes.mat")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    sorting, sorting_tree, meta = convert_mat_to_sortings(str(data_file))
    waveforms = meta["waveforms"]
    spike_times_all = meta["spiketimes"]
    fs = meta["Fs"]
    hierarchy_tree = meta["hierarchy_tree"]
    gt_assigns = meta.get("curation_assigns")

    manager = ClusterManager(
        initial_assigns=meta["hierarchy_assigns"],
        overcluster_assigns=meta["overcluster_assigns"],
        hierarchy_tree=hierarchy_tree,
        spike_times=spike_times_all,
        waveforms=waveforms,
    )
    features = ClusterFeatures(meta=meta, assigns=manager.assigns)

    pipeline = PureVLMCurationPipeline(
        manager=manager,
        features=features,
        sampling_rate=fs,
        auto_discard_threshold=args.auto_discard_threshold,
        small_cluster_threshold=args.small_cluster_threshold,
        final_minimum_threshold=args.final_minimum_threshold,
        provider=args.provider,
        model=model,
        use_mock=args.use_mock,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        output_dir=channel_output,
    )

    final_clusters = pipeline.run_full_pipeline()
    pipeline.save_action_log(channel_output / "action_log.csv")
    np.save(channel_output / "final_assigns.npy", manager.assigns)
    np.save(channel_output / "final_hierarchy_tree.npy", manager.hierarchy_tree)
    np.save(channel_output / "overcluster_assigns.npy", manager.overcluster_assigns)

    if final_clusters:
        spike_frames = (spike_times_all * fs).astype(np.int64)
        final_sorting = si.NumpySorting.from_unit_dict(
            {int(cid): spike_frames[manager.assigns == cid] for cid in final_clusters},
            sampling_frequency=fs,
        )
        gt_sorting = None
        if gt_assigns is not None:
            gt_cluster_ids = np.unique(gt_assigns[gt_assigns > 0])
            gt_sorting = si.NumpySorting.from_unit_dict(
                {int(cid): spike_frames[gt_assigns == cid] for cid in gt_cluster_ids},
                sampling_frequency=fs,
            )
        report = generate_full_evaluation_report(
            curated_sorting=final_sorting,
            waveforms=waveforms,
            spike_times=spike_times_all,
            assigns=manager.assigns,
            ground_truth_sorting=gt_sorting,
            gt_assigns=gt_assigns,
            sampling_frequency=fs,
            output_dir=channel_output,
        )
        print_evaluation_summary(report)
    else:
        print("[Warning] No final clusters - skipped evaluation.")

    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Done. Results: {channel_output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Run failed: {e}")
        sys.exit(1)
