"""
Unit-test style evaluation on dataset prompts+images (not full pipeline replay).

Goal:
- Feed each sample's prompt + images to model
- Parse predicted action
- Compare against human action label
- Report per-channel and per-action accuracy (SPLIT/MERGE/DISCARD)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agent.api import call_vlm
from src.agent.runner import _sanitize_json_response


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

SPLIT_ACTIONS = {"KEEP", "DISCARD", "SPLIT"}
MERGE_ACTIONS = {"MERGE", "NOT_MERGE", "DISCARD"}

SPLIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(SPLIT_ACTIONS)},
        "rationale": {"type": "string"},
    },
    "required": ["action", "rationale"],
    "additionalProperties": False,
}

MERGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(MERGE_ACTIONS)},
        "rationale": {"type": "string"},
    },
    "required": ["action", "rationale"],
    "additionalProperties": False,
}

SPLIT_ACTION_ONLY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(SPLIT_ACTIONS)},
    },
    "required": ["action"],
    "additionalProperties": True,
}

MERGE_ACTION_ONLY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(MERGE_ACTIONS)},
    },
    "required": ["action"],
    "additionalProperties": True,
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_channels(raw: str) -> List[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _resolve_image(dataset_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (dataset_root / p).resolve()


def _image_b64s(row: Dict[str, Any], dataset_root: Path) -> List[str]:
    stage = row.get("stage")
    keys = SPLIT_IMAGE_KEYS if stage == "split" else MERGE_IMAGE_KEYS
    images = row.get("images", {}) or {}
    out: List[str] = []
    for key in keys:
        rel = images.get(key)
        if not rel:
            continue
        path = _resolve_image(dataset_root, rel)
        if not path.exists():
            continue
        out.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
    return out


def _allowed_actions_for_row(row: Dict[str, Any]) -> List[str]:
    allowed = row.get("allowed_actions")
    if isinstance(allowed, list) and allowed:
        return [str(x).upper() for x in allowed]
    stage = str(row.get("stage", "split"))
    return sorted(SPLIT_ACTIONS if stage == "split" else MERGE_ACTIONS)


def _action_only_instruction(stage: str, allowed_actions: Sequence[str]) -> str:
    choices = "|".join(allowed_actions)
    return (
        "\n\nFinal instruction:\n"
        f"- Output exactly one token from: {choices}\n"
        "- Do not output rationale, explanation, JSON, markdown, or extra text."
    )


def _predict_action(
    *,
    prompt: str,
    images_b64: List[str],
    provider: str,
    model: str,
    stage: str,
    allowed_actions: Sequence[str],
    enable_thinking: bool,
    temperature: float,
    reasoning_effort: Optional[str],
    use_response_schema: bool,
    enforce_action_only: bool,
    max_tokens: int,
) -> Tuple[str, str]:
    kwargs: Dict[str, Any] = {}
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    if use_response_schema:
        if enforce_action_only:
            kwargs["response_schema"] = (
                SPLIT_ACTION_ONLY_SCHEMA if stage == "split" else MERGE_ACTION_ONLY_SCHEMA
            )
        else:
            kwargs["response_schema"] = SPLIT_SCHEMA if stage == "split" else MERGE_SCHEMA
    if enforce_action_only:
        prompt = prompt + _action_only_instruction(stage, allowed_actions)
    raw = call_vlm(
        prompt=prompt,
        images=images_b64,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )
    return _extract_action(raw, allowed_actions=allowed_actions), raw


def _extract_action(raw: str, allowed_actions: Sequence[str]) -> str:
    allowed = {a.upper() for a in allowed_actions}
    text = (raw or "").strip()
    upper_text = text.upper()

    # 0) Action-only ideal case: single token output.
    if upper_text in allowed:
        return upper_text

    # 1) JSON
    try:
        decision = json.loads(_sanitize_json_response(text))
        action = str(decision.get("action", "")).upper().strip()
        if action in allowed:
            return action
    except Exception:
        pass

    # 2) Common explicit action fields/tags
    # Natural-language target format: Action: XXX
    m = re.search(r"Action\s*:\s*([A-Z_]+)", raw, flags=re.IGNORECASE)
    if m:
        action = m.group(1).upper().strip()
        if action in allowed:
            return action
    # XML-ish tags: <action>XXX</action>
    m = re.search(r"<\s*action\s*>\s*([A-Z_]+)\s*<\s*/\s*action\s*>", raw, flags=re.IGNORECASE)
    if m:
        action = m.group(1).upper().strip()
        if action in allowed:
            return action

    # 3) End-of-response fallback: only trust the tail section, not whole reasoning.
    tail = "\n".join(text.splitlines()[-8:]).upper()
    # a) explicit FINAL/ANSWER/ACTION line near the end
    for pat in [
        r"(?:FINAL\s+ANSWER|FINAL\s+ACTION|ANSWER|ACTION)\s*[:：]\s*([A-Z_]+)",
        r"^\s*([A-Z_]+)\s*$",
    ]:
        m = re.search(pat, tail, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            action = m.group(1).upper().strip()
            if action in allowed:
                return action
            if action in {"NOT_MERGE", "MERGE", "SPLIT", "DISCARD", "KEEP"}:
                return "INVALID_ACTION"

    # 4) If only disallowed action appears in tail, mark invalid; otherwise parse error.
    if re.search(r"\b(NOT_MERGE|MERGE|SPLIT|DISCARD|KEEP)\b", tail):
        return "INVALID_ACTION"
    return "PARSE_ERROR"


def _is_human_action_sample(row: Dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    action = str(row.get("label_action", "")).upper()
    return source.startswith("expert_") and action in {"SPLIT", "MERGE", "DISCARD"}


def _compact_log_text(text: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action reproduction from finetune dataset.")
    parser.add_argument("--input-jsonl", default="output/finetune_dataset/finetune_dataset_mixed.jsonl")
    parser.add_argument("--dataset-root", default="output/finetune_dataset")
    parser.add_argument("--eval-channels", default="CH31")
    parser.add_argument("--provider", default="vllm", choices=["gpt4o", "openrouter", "vllm", "claude"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="output/unit_test_dataset_eval")
    parser.add_argument(
        "--raw-log-file",
        default="",
        help="Optional path to append full raw LLM outputs for debugging.",
    )
    parser.add_argument(
        "--print-first-raw",
        action="store_true",
        help="Print full raw output for the first eval sample to stdout.",
    )
    parser.add_argument("--enforce-action-only", action="store_true", default=True)
    parser.add_argument("--allow-reasoned-output", action="store_true")
    parser.add_argument(
        "--use-response-schema",
        action="store_true",
        help="Force structured output schema in API call; off by default to respect prompt format",
    )
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enforce_action_only = args.enforce_action_only and not args.allow_reasoned_output
    enable_thinking = bool(args.enable_thinking) and not bool(args.disable_thinking)

    channels = set(_parse_channels(args.eval_channels))
    rows = _read_jsonl(input_jsonl)
    rows = [
        r for r in rows
        if str(r.get("channel", "")).upper() in channels and _is_human_action_sample(r)
    ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No eval rows after filtering")

    results: List[Dict[str, Any]] = []
    raw_log_path = Path(args.raw_log_file) if args.raw_log_file else None
    if raw_log_path:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_log_path, "w") as f:
            f.write("")

    for i, row in enumerate(rows, 1):
        gt = str(row["label_action"]).upper()
        stage = str(row.get("stage", "split"))
        allowed_actions = _allowed_actions_for_row(row)
        imgs = _image_b64s(row, dataset_root=dataset_root)
        if not imgs:
            pred = "MISSING_IMAGE"
            raw = ""
        else:
            pred, raw = _predict_action(
                prompt=row["prompt"],
                images_b64=imgs,
                provider=args.provider,
                model=args.model,
                stage=stage,
                allowed_actions=allowed_actions,
                enable_thinking=enable_thinking,
                temperature=args.temperature,
                reasoning_effort=args.reasoning_effort,
                use_response_schema=args.use_response_schema,
                enforce_action_only=enforce_action_only,
                max_tokens=args.max_tokens,
            )

        results.append(
            {
                "idx": i,
                "id": row.get("id"),
                "channel": row.get("channel"),
                "stage": stage,
                "source": row.get("source"),
                "profile": row.get("profile"),
                "output_mode": row.get("output_mode"),
                "allowed_actions": ",".join(allowed_actions),
                "gt_action": gt,
                "pred_action": pred,
                "match": int(pred == gt),
                "raw_response": raw,
            }
        )
        line = f"[{i}/{len(rows)}] {row.get('id')} gt={gt} pred={pred} match={pred == gt}"
        if pred in {"INVALID_ACTION", "PARSE_ERROR"}:
            line += f" (llm_raw='{_compact_log_text(raw)}')"
        print(line, flush=True)
        if args.print_first_raw and i == 1:
            print("===== FIRST RAW BEGIN =====", flush=True)
            print(raw if raw is not None else "", flush=True)
            print("===== FIRST RAW END =====", flush=True)
        if raw_log_path is not None:
            with open(raw_log_path, "a") as f:
                f.write(f"[{i}/{len(rows)}] id={row.get('id')} channel={row.get('channel')} stage={stage}\n")
                f.write(f"gt={gt} pred={pred} match={pred == gt}\n")
                f.write("----- RAW BEGIN -----\n")
                f.write(raw if raw is not None else "")
                f.write("\n----- RAW END -----\n\n")

    detail_csv = output_dir / "detail_predictions.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Aggregate per channel and gt action
    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (str(r["channel"]), str(r["gt_action"]))
        grouped.setdefault(key, []).append(r)

    for (ch, gt_action), items in sorted(grouped.items()):
        n = len(items)
        correct = sum(int(x["match"]) for x in items)
        summary_rows.append(
            {
                "channel": ch,
                "gt_action": gt_action,
                "n": n,
                "correct": correct,
                "accuracy": round(correct / n, 4) if n else 0.0,
            }
        )

    # Also add channel-level overall
    by_channel: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_channel.setdefault(str(r["channel"]), []).append(r)
    for ch, items in sorted(by_channel.items()):
        n = len(items)
        correct = sum(int(x["match"]) for x in items)
        summary_rows.append(
            {
                "channel": ch,
                "gt_action": "ALL",
                "n": n,
                "correct": correct,
                "accuracy": round(correct / n, 4) if n else 0.0,
            }
        )

    summary_csv = output_dir / "summary_accuracy.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "gt_action", "n", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(output_dir / "summary_accuracy.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    run_manifest = {
        "timestamp_s": int(time.time()),
        "input_jsonl": str(input_jsonl),
        "dataset_root": str(dataset_root),
        "provider": args.provider,
        "model": args.model,
        "eval_channels": sorted(channels),
        "n_samples": len(results),
        "use_response_schema": args.use_response_schema,
        "enforce_action_only": enforce_action_only,
        "enable_thinking": enable_thinking,
        "max_tokens": args.max_tokens,
        "vlm_extra_body_json": os.getenv("VLM_EXTRA_BODY_JSON", ""),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Saved detail: {detail_csv}", flush=True)
    print(f"Saved summary: {summary_csv}", flush=True)
    print(f"Saved manifest: {output_dir / 'run_manifest.json'}", flush=True)
    if raw_log_path is not None:
        print(f"Saved raw log: {raw_log_path}", flush=True)


if __name__ == "__main__":
    main()
