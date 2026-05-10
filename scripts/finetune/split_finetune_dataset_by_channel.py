"""
Create train/eval JSONL splits by channel from finetune_dataset_mixed.jsonl.

Defaults:
- train channels: CH3, CH20, CH30
- eval channel: CH31
- expert-only rows (human actions only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_channels(raw: str) -> List[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _filter_rows(rows: Sequence[Dict[str, Any]], channels: Sequence[str], expert_only: bool) -> List[Dict[str, Any]]:
    wanted = set(channels)
    out: List[Dict[str, Any]] = []
    for row in rows:
        ch = str(row.get("channel", "")).upper()
        if ch not in wanted:
            continue
        if expert_only and not str(row.get("source", "")).startswith("expert_"):
            continue
        out.append(row)
    return out


def _count_actions(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        a = str(row.get("label_action", ""))
        out[a] = out.get(a, 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Split finetune dataset by channels.")
    parser.add_argument("--input-jsonl", default="output/finetune_dataset/finetune_dataset_mixed.jsonl")
    parser.add_argument("--output-dir", default="output/finetune_dataset_splits")
    parser.add_argument("--train-channels", default="CH3,CH20,CH30")
    parser.add_argument("--eval-channels", default="CH31")
    parser.add_argument("--expert-only", action="store_true", default=True)
    parser.add_argument("--include-synthetic", action="store_true")
    args = parser.parse_args()

    expert_only = args.expert_only and not args.include_synthetic
    rows = _read_jsonl(Path(args.input_jsonl))
    train_channels = _parse_channels(args.train_channels)
    eval_channels = _parse_channels(args.eval_channels)

    train_rows = _filter_rows(rows, train_channels, expert_only=expert_only)
    eval_rows = _filter_rows(rows, eval_channels, expert_only=expert_only)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    summary = {
        "input_jsonl": args.input_jsonl,
        "expert_only": expert_only,
        "train_channels": train_channels,
        "eval_channels": eval_channels,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_actions": _count_actions(train_rows),
        "eval_actions": _count_actions(eval_rows),
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
    }
    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
