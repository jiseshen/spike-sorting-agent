"""
Vision fine-tuning for Qwen3.5-4B with Unsloth, aligned to Unsloth Vision notebook.

Key alignment choices:
- Uses FastVisionModel + UnslothVisionDataCollator
- Trains on multimodal "messages" format (text prompt + multiple images)
- Keeps output format as JSON string: {"action":"...","rationale":"..."}

Default split:
- Train channels: CH3, CH20, CH30
- Eval channel: CH31
- Expert-only samples (human actions): source starts with "expert_"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image


SPLIT_IMAGE_KEYS: Sequence[str] = (
    "waveform_overlay",
    "isi_histogram",
    "amplitude_distribution",
    "aggregation_tree",
)
MERGE_IMAGE_KEYS: Sequence[str] = (
    "small_waveform_overlay",
    "large_waveform_overlay",
    "merged_isi_histogram",
)


def _read_jsonl(path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _parse_channels(raw: str) -> List[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _resolve_image_path(dataset_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (dataset_root / p).resolve()


def _row_images(row: Dict[str, Any], dataset_root: Path) -> List[Image.Image]:
    stage = row.get("stage")
    images = row.get("images", {}) or {}
    keys = SPLIT_IMAGE_KEYS if stage == "split" else MERGE_IMAGE_KEYS

    loaded: List[Image.Image] = []
    for key in keys:
        rel = images.get(key)
        if not rel:
            continue
        path = _resolve_image_path(dataset_root, rel)
        if not path.exists():
            continue
        loaded.append(Image.open(path).convert("RGB"))
    return loaded


def _assistant_target(row: Dict[str, Any], target_mode: str = "original") -> str:
    if target_mode == "original":
        t = str(row.get("target", "")).strip()
        if t:
            return t
    action = str(row.get("label_action", "")).strip().upper()
    rationale = str(row.get("label_reason", "")).strip()
    if not rationale:
        rationale = "No rationale provided."
    return json.dumps({"action": action, "rationale": rationale}, ensure_ascii=False)


def _to_messages(row: Dict[str, Any], dataset_root: Path, target_mode: str = "original") -> Dict[str, Any]:
    prompt = str(row["prompt"])
    imgs = _row_images(row, dataset_root=dataset_root)
    if not imgs:
        raise ValueError(f"No valid images for sample {row.get('id')}")

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in imgs:
        user_content.append({"type": "image", "image": img})

    return {
        "id": row.get("id"),
        "channel": row.get("channel"),
        "label_action": row.get("label_action"),
        "target_format": row.get("target_format"),
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _assistant_target(row, target_mode=target_mode)}],
            },
        ],
    }


def _filter_rows(
    rows: List[Dict[str, Any]],
    channels: Sequence[str],
    expert_only: bool,
) -> List[Dict[str, Any]]:
    keep = []
    channel_set = {c.upper() for c in channels}
    for row in rows:
        if str(row.get("channel", "")).upper() not in channel_set:
            continue
        if expert_only and not str(row.get("source", "")).startswith("expert_"):
            continue
        if not row.get("prompt"):
            continue
        if not row.get("label_action"):
            continue
        keep.append(row)
    return keep


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_ch: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    for r in rows:
        ch = str(r.get("channel"))
        act = str(r.get("label_action"))
        by_ch[ch] = by_ch.get(ch, 0) + 1
        by_action[act] = by_action.get(act, 0) + 1
    return {"by_channel": by_ch, "by_action": by_action}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision SFT for Qwen3.5-4B on SpikeSorting dataset.")
    parser.add_argument("--input-jsonl", default="output/finetune_dataset/finetune_dataset_mixed.jsonl")
    parser.add_argument("--dataset-root", default="output/finetune_dataset")
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--output-dir", default="output/finetune_qwen35_4b_vision_lora")
    parser.add_argument("--train-channels", default="CH3,CH20,CH30")
    parser.add_argument("--eval-channels", default="CH31")
    parser.add_argument("--expert-only", action="store_true", default=True)
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--target-mode",
        default="original",
        choices=["original", "json_action"],
        help="original=use dataset target verbatim; json_action=force JSON action/rationale target",
    )
    args = parser.parse_args()

    try:
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTConfig, SFTTrainer
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for vision SFT. Install Unsloth vision stack first."
        ) from e

    input_jsonl = Path(args.input_jsonl)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    train_channels = _parse_channels(args.train_channels)
    eval_channels = _parse_channels(args.eval_channels)
    expert_only = args.expert_only and not args.include_synthetic

    all_rows = _read_jsonl(input_jsonl, max_rows=args.max_rows)
    train_rows = _filter_rows(all_rows, channels=train_channels, expert_only=expert_only)
    eval_rows = _filter_rows(all_rows, channels=eval_channels, expert_only=expert_only)
    if not train_rows:
        raise ValueError("No train samples after filtering")

    train_dataset = [_to_messages(r, dataset_root=dataset_root, target_mode=args.target_mode) for r in train_rows]
    eval_dataset = (
        [_to_messages(r, dataset_root=dataset_root, target_mode=args.target_mode) for r in eval_rows]
        if eval_rows
        else None
    )

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
        ),
    )
    stats = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = output_dir / "merged_16bit"
    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except Exception as e:
        print(f"[Warn] Skipped merged save: {e}")

    summary = {
        "input_jsonl": str(input_jsonl),
        "dataset_root": str(dataset_root),
        "model_name": args.model_name,
        "load_in_4bit": args.load_in_4bit,
        "train_channels": train_channels,
        "eval_channels": eval_channels,
        "expert_only": expert_only,
        "n_train": len(train_dataset),
        "n_eval": len(eval_dataset) if eval_dataset is not None else 0,
        "train_distribution": _summarize(train_rows),
        "eval_distribution": _summarize(eval_rows),
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_mode": args.target_mode,
    }
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved adapter to: {adapter_dir}")
    print(f"Train rows: {len(train_dataset)} | Eval rows: {summary['n_eval']}")


if __name__ == "__main__":
    main()
