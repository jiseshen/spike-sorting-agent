"""
SFT adapter: convert adapt/sampler output into HuggingFace format and
optionally launch a training script.

This module bridges between the project's jsonl format and the
existing scripts/finetune/ training scripts (train_qwen35_unsloth.py,
train_gemma4_unsloth.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def prepare_hf_dataset(
    adapter_dir: str | Path,
    output_subdir: str = "hf_dataset",
) -> Path:
    """
    Convert train_dataset.jsonl (from sampler.py) into HuggingFace
    datasets format (instruction / input / output columns).

    Each example:
      instruction: "Spike sorting curation decision. Action phase: <phase>."
      input: "Cluster <id>. GT reasoning: <reasoning>. Teacher feedback: <feedback>."
      output: "<gt_action>"

    Args:
        adapter_dir: Path to the adapter run directory.
        output_subdir: Subdirectory name for the HF dataset files.

    Returns:
        Path to the HF dataset directory.
    """
    adapter_dir = Path(adapter_dir)
    src_file = adapter_dir / "train_dataset.jsonl"
    hf_dir = adapter_dir / output_subdir
    hf_dir.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    with open(src_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            instruction = f"Spike sorting curation decision. Action phase: {d['action_phase']}."
            input_text = (
                f"Cluster {d['cluster_id']}."
                + (f" Merge target: {d['target_id']}." if d.get("target_id") else "")
                + f" Teacher feedback: {d['teacher_feedback']}"
            )
            output_text = d["gt_action"]
            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "gt_reasoning": d["gt_reasoning"],
            })

    out_file = hf_dir / "train.jsonl"
    with open(out_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    meta = {"n_examples": len(examples), "source": str(src_file)}
    with open(hf_dir / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [done] HF dataset: {len(examples)} examples → {hf_dir}")
    return hf_dir


def launch_training(
    adapter_dir: str | Path,
    adapter_name: str = "qwen35",
    hf_subdir: str = "hf_dataset",
    extra_args: Optional[list[str]] = None,
) -> int:
    """
    Launch a finetune training script for the prepared HF dataset.

    Maps adapter_name to the corresponding script in scripts/finetune/.

    Args:
        adapter_dir: Path to the adapter run directory.
        adapter_name: Which model to finetune ("qwen35" or "gemma4").
        hf_subdir: Subdirectory containing the HF dataset.
        extra_args: Additional CLI args passed to the training script.

    Returns:
        Return code from the training subprocess.
    """
    adapter_dir = Path(adapter_dir)
    hf_dir = adapter_dir / hf_subdir

    script_map = {
        "qwen35": "scripts/finetune/train_qwen35_unsloth.py",
        "gemma4": "scripts/finetune/train_gemma4_unsloth.py",
    }
    script = script_map.get(adapter_name)
    if script is None:
        raise ValueError(f"Unknown adapter '{adapter_name}'. Choose from: {list(script_map)}")

    cmd = [
        sys.executable,
        script,
        "--dataset-dir", str(hf_dir),
        "--output-dir", str(adapter_dir / "model_checkpoint"),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [launch] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode
