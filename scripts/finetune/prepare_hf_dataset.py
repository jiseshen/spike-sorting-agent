"""
Convert finetune JSONL + images into Hugging Face datasets format.

Features:
- Reads finetune_dataset_mixed.jsonl
- Flattens metrics/images into stable columns
- Casts image columns to `datasets.Image` feature
- Saves as `datasets` artifact via `save_to_disk`
- Optional push to Hugging Face Hub private dataset repo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


IMAGE_COLUMNS = [
    "image_waveform_overlay",
    "image_isi_histogram",
    "image_amplitude_distribution",
    "image_aggregation_tree",
    "image_small_waveform_overlay",
    "image_large_waveform_overlay",
    "image_merged_isi_histogram",
]

METRIC_COLUMNS = [
    "n_spikes",
    "n_overclusters",
    "isi_violation_rate",
    "amplitude_cv",
    "small_n_spikes",
    "large_n_spikes",
    "small_isi_rate",
    "large_isi_rate",
    "waveform_correlation",
    "merged_isi_rate",
]


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


def _resolve_image_path(dataset_root: Path, rel_or_abs: Optional[str]) -> Optional[str]:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str((dataset_root / p).resolve())


def _normalize_records(raw_rows: List[Dict[str, Any]], dataset_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in raw_rows:
        metrics = row.get("metrics", {}) or {}
        images = row.get("images", {}) or {}

        rec: Dict[str, Any] = {
            "id": row.get("id"),
            "channel": row.get("channel"),
            "stage": row.get("stage"),
            "source": row.get("source"),
            "expert_action_raw": row.get("expert_action_raw"),
            "prompt": row.get("prompt"),
            "target": row.get("target"),
            "target_format": row.get("target_format"),
            "label_action": row.get("label_action"),
            "label_reason": row.get("label_reason"),
            "small_cluster_id": row.get("small_cluster_id"),
            "large_cluster_id": row.get("large_cluster_id"),
            "metrics_json": json.dumps(metrics, ensure_ascii=False),
        }

        for k in METRIC_COLUMNS:
            rec[k] = metrics.get(k)

        rec["image_waveform_overlay"] = _resolve_image_path(dataset_root, images.get("waveform_overlay"))
        rec["image_isi_histogram"] = _resolve_image_path(dataset_root, images.get("isi_histogram"))
        rec["image_amplitude_distribution"] = _resolve_image_path(
            dataset_root, images.get("amplitude_distribution")
        )
        rec["image_aggregation_tree"] = _resolve_image_path(dataset_root, images.get("aggregation_tree"))
        rec["image_small_waveform_overlay"] = _resolve_image_path(
            dataset_root, images.get("small_waveform_overlay")
        )
        rec["image_large_waveform_overlay"] = _resolve_image_path(
            dataset_root, images.get("large_waveform_overlay")
        )
        rec["image_merged_isi_histogram"] = _resolve_image_path(
            dataset_root, images.get("merged_isi_histogram")
        )

        records.append(rec)

    return records


def build_dataset(
    records: List[Dict[str, Any]],
):
    from datasets import Dataset, Features, Image, Value

    features = Features(
        {
            "id": Value("string"),
            "channel": Value("string"),
            "stage": Value("string"),
            "source": Value("string"),
            "expert_action_raw": Value("string"),
            "prompt": Value("string"),
            "target": Value("string"),
            "target_format": Value("string"),
            "label_action": Value("string"),
            "label_reason": Value("string"),
            "small_cluster_id": Value("int64"),
            "large_cluster_id": Value("int64"),
            "metrics_json": Value("string"),
            "n_spikes": Value("int64"),
            "n_overclusters": Value("int64"),
            "isi_violation_rate": Value("float64"),
            "amplitude_cv": Value("float64"),
            "small_n_spikes": Value("int64"),
            "large_n_spikes": Value("int64"),
            "small_isi_rate": Value("float64"),
            "large_isi_rate": Value("float64"),
            "waveform_correlation": Value("float64"),
            "merged_isi_rate": Value("float64"),
            "image_waveform_overlay": Image(),
            "image_isi_histogram": Image(),
            "image_amplitude_distribution": Image(),
            "image_aggregation_tree": Image(),
            "image_small_waveform_overlay": Image(),
            "image_large_waveform_overlay": Image(),
            "image_merged_isi_histogram": Image(),
        }
    )
    return Dataset.from_list(records, features=features)


def _coerce_nulls(records: List[Dict[str, Any]]) -> None:
    int_cols = ["small_cluster_id", "large_cluster_id", "n_spikes", "n_overclusters", "small_n_spikes", "large_n_spikes"]
    float_cols = ["isi_violation_rate", "amplitude_cv", "small_isi_rate", "large_isi_rate", "waveform_correlation", "merged_isi_rate"]

    for rec in records:
        for k in int_cols:
            if rec.get(k) is None:
                rec[k] = -1
        for k in float_cols:
            if rec.get(k) is None:
                rec[k] = float("nan")
        for k in IMAGE_COLUMNS:
            if rec.get(k) is None:
                rec[k] = None
        for k in ["expert_action_raw"]:
            if rec.get(k) is None:
                rec[k] = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hugging Face datasets format from finetune JSONL.")
    parser.add_argument(
        "--input-jsonl",
        default="output/finetune_dataset/finetune_dataset_mixed.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        default="output/finetune_dataset",
        help="Root directory that contains images/ and summary files.",
    )
    parser.add_argument(
        "--save-dir",
        default="output/finetune_dataset_hf",
        help="Path for datasets.save_to_disk output.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Debug only. 0 means all rows.")
    parser.add_argument("--repo-id", default="", help="Optional: HF dataset repo id like user/repo.")
    parser.add_argument("--private", action="store_true", help="When pushing, create/update as private dataset.")
    parser.add_argument("--token", default="", help="Optional HF token. If empty, uses CLI login/env.")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    dataset_root = Path(args.dataset_root)
    save_dir = Path(args.save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    raw_rows = _read_jsonl(input_jsonl, max_rows=args.max_rows)
    if not raw_rows:
        raise ValueError(f"No rows loaded from {input_jsonl}")

    records = _normalize_records(raw_rows, dataset_root=dataset_root)
    _coerce_nulls(records)

    ds = build_dataset(records)
    ds.save_to_disk(str(save_dir))

    print(f"Saved datasets artifact: {save_dir}")
    print(f"Rows: {len(ds)}")
    print(f"Columns: {len(ds.column_names)}")

    if args.repo_id:
        from datasets import DatasetDict

        ds_dict = DatasetDict({"train": ds})
        ds_dict.push_to_hub(
            repo_id=args.repo_id,
            private=args.private,
            token=(args.token or None),
        )
        print(f"Pushed to HF Hub: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

