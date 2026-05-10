"""
One-shot demo: simulated human feedback on a single random action-sheet step.

Workflow:
1) Randomly sample a channel + step
2) Replay all prior ground-truth actions to reconstruct identical state
3) Query student model (gpt-5.4) for action + rationale
4) Query teacher model (gpt-5.4) for brief feedback using:
   - same student observation context (images + metrics; no student instructions)
   - student action + rationale
   - hidden reference annotation for calibration
"""

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.cluster.auto_filter import automatic_size_filter
from src.cluster.manager import ClusterManager
from src.io.matlab_loader import load_matlab_spikes
from src.agent.teacher_feedback import build_teacher_feedback_prompt
from src.agent.api import get_call_history
from src.agent.runner import (
    call_vlm_api,
    compute_merged_isi_violation_rate,
    compute_waveform_correlation,
    create_aggregation_tree_image,
    create_isi_histogram_image,
    create_waveform_overlay_image,
    vlm_phase1_cluster_decision,
    vlm_phase2_merge_decision,
)

CHANNELS = ["CH3", "CH20", "CH30", "CH31"]
AUTO_DISCARD_THRESHOLD = 500
PROVIDER = "gpt4o"
STUDENT_MODEL = "gpt-5.4"
TEACHER_MODEL = "gpt-5.4"
TEACHER_PROVIDER = "gpt4o"
USE_MOCK = False
TEMPERATURE = 0.0
REASONING_EFFORT = "none"
OUTPUT_BASE = Path("output/simulated_teacher_demo")


@dataclass
class DemoSelection:
    channel: str
    step: int
    action_type: str
    cluster_id: int
    target_id: Optional[int]
    action_str: str
    human_reasoning: str
    manager: ClusterManager
    Fs: float


def parse_action(action_str: str) -> Tuple[str, int, Optional[int]]:
    action_str = action_str.strip().strip("'\"")
    split_match = re.match(r"s\s+(\d+)", action_str)
    if split_match:
        return "split", int(split_match.group(1)), None

    merge_match = re.match(r"m\s+(\d+)\s+(\d+)", action_str)
    if merge_match:
        target = int(merge_match.group(1))
        source = int(merge_match.group(2))
        if target == 0:
            return "discard", source, None
        return "merge", source, target

    raise ValueError(f"Cannot parse action: {action_str}")


def load_action_sheet(channel: str) -> List[Tuple[str, str]]:
    csv_path = Path(f"data/action_sheets/{channel}.csv")
    actions: List[Tuple[str, str]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append((row["Actions"].strip(), row.get("Action Reasoning", "").strip()))
    return actions


def setup_manager(channel: str) -> Tuple[ClusterManager, Dict, float]:
    data = load_matlab_spikes(f"data/{channel}_spikes.mat")
    manager = ClusterManager(
        initial_assigns=data["hierarchy_assigns"],
        overcluster_assigns=data["overcluster_assigns"],
        hierarchy_tree=data["hierarchy_tree"],
        spike_times=data["spiketimes"],
        waveforms=data["waveforms"],
    )

    _, discarded_actions = automatic_size_filter(manager.assigns, AUTO_DISCARD_THRESHOLD)
    for action in discarded_actions:
        manager.discard_cluster(action.cluster_id)

    return manager, data, data["Fs"]


def execute_human_action(
    manager: ClusterManager,
    action_type: str,
    cluster_id: int,
    target_id: Optional[int],
) -> None:
    if action_type == "split":
        try:
            manager.split_last_merge(cluster_id)
        except Exception:
            pass
    elif action_type == "discard":
        manager.discard_cluster(cluster_id)
    elif action_type == "merge" and target_id is not None:
        manager.merge_clusters([cluster_id, target_id], target_id=target_id)


def sample_valid_step(seed: Optional[int] = None, max_attempts: int = 100) -> DemoSelection:
    rng = random.Random(seed)
    available_channels = [
        ch for ch in CHANNELS if Path(f"data/{ch}_spikes.mat").exists() and Path(f"data/action_sheets/{ch}.csv").exists()
    ]
    if not available_channels:
        raise RuntimeError("No channels have both .mat data and action sheet files.")

    for _ in range(max_attempts):
        channel = rng.choice(available_channels)
        action_sheet = load_action_sheet(channel)
        if not action_sheet:
            continue
        step = rng.randint(1, len(action_sheet))
        action_str, human_reasoning = action_sheet[step - 1]

        try:
            action_type, cluster_id, target_id = parse_action(action_str)
        except ValueError:
            continue

        manager, data, Fs = setup_manager(channel)

        # Replay all previous GT actions to reconstruct the exact state at sampled step
        for prev_action_str, _ in action_sheet[: step - 1]:
            try:
                prev_type, prev_cluster, prev_target = parse_action(prev_action_str)
            except ValueError:
                continue
            execute_human_action(manager, prev_type, prev_cluster, prev_target)

        info = manager.get_cluster_info(cluster_id)
        if info is None or info["n_spikes"] == 0:
            continue

        if action_type == "merge":
            target_info = manager.get_cluster_info(target_id)
            if target_info is None or target_info["n_spikes"] == 0:
                continue

        return DemoSelection(
            channel=channel,
            step=step,
            action_type=action_type,
            cluster_id=cluster_id,
            target_id=target_id,
            action_str=action_str,
            human_reasoning=human_reasoning,
            manager=manager,
            Fs=Fs,
        )

    raise RuntimeError("Failed to sample a valid step after repeated attempts.")


def run_demo(
    seed: Optional[int] = None,
    student_model: str = STUDENT_MODEL,
    teacher_model: str = TEACHER_MODEL,
    teacher_provider: Optional[str] = None,
) -> Dict[str, object]:
    selection = sample_valid_step(seed=seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = OUTPUT_BASE / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_teacher_provider = teacher_provider or (
        "claude" if teacher_model.lower().startswith("claude") else TEACHER_PROVIDER
    )

    human_action = {
        "split": "SPLIT",
        "discard": "DISCARD",
        "merge": "MERGE",
    }[selection.action_type]

    history_before_student = len(get_call_history())

    if selection.action_type in {"split", "discard"}:
        info = selection.manager.get_cluster_info(selection.cluster_id)
        student_resp = vlm_phase1_cluster_decision(
            cluster_id=selection.cluster_id,
            waveforms=info["waveforms"],
            spike_times=info["spike_times"],
            overcluster_composition=info["overclusters"],
            hierarchy_tree=selection.manager.hierarchy_tree,
            sampling_rate=selection.Fs,
            provider=PROVIDER,
            model=student_model,
            use_mock=USE_MOCK,
            temperature=TEMPERATURE,
            reasoning_effort=REASONING_EFFORT,
            output_dir=out_dir / "student",
        )

        teacher_images = [
            create_waveform_overlay_image(info["waveforms"], selection.cluster_id, selection.Fs),
            create_isi_histogram_image(info["spike_times"], selection.cluster_id),
            create_aggregation_tree_image(
                selection.manager.hierarchy_tree,
                info["overclusters"],
                selection.cluster_id,
            ),
        ]
        teacher_context = {
            "cluster_id": selection.cluster_id,
            "n_spikes": int(info["n_spikes"]),
            "n_overclusters": len(info["overclusters"]),
        }
    else:
        small_info = selection.manager.get_cluster_info(selection.cluster_id)
        large_info = selection.manager.get_cluster_info(selection.target_id)
        student_resp = vlm_phase2_merge_decision(
            small_cluster_id=selection.cluster_id,
            small_waveforms=small_info["waveforms"],
            small_spike_times=small_info["spike_times"],
            large_cluster_id=selection.target_id,
            large_waveforms=large_info["waveforms"],
            large_spike_times=large_info["spike_times"],
            sampling_rate=selection.Fs,
            provider=PROVIDER,
            model=student_model,
            use_mock=USE_MOCK,
            temperature=TEMPERATURE,
            reasoning_effort=REASONING_EFFORT,
            output_dir=out_dir / "student",
        )

        corr = compute_waveform_correlation(small_info["waveforms"], large_info["waveforms"])
        merged_isi_rate = compute_merged_isi_violation_rate(
            small_info["spike_times"], large_info["spike_times"]
        )
        small_isi_rate = compute_merged_isi_violation_rate(
            small_info["spike_times"], small_info["spike_times"][:1]
        )
        large_isi_rate = compute_merged_isi_violation_rate(
            large_info["spike_times"], large_info["spike_times"][:1]
        )
        teacher_images = [
            create_waveform_overlay_image(small_info["waveforms"], selection.cluster_id, selection.Fs),
            create_waveform_overlay_image(large_info["waveforms"], selection.target_id, selection.Fs),
            create_isi_histogram_image(
                np.sort(
                    np.concatenate([small_info["spike_times"], large_info["spike_times"]])
                ),
                f"{selection.cluster_id}+{selection.target_id}",
            ),
        ]
        teacher_context = {
            "cluster_id": selection.cluster_id,
            "merge_target_id": int(selection.target_id),
            "n_small": int(small_info["n_spikes"]),
            "n_large": int(large_info["n_spikes"]),
            "small_isi_rate": float(small_isi_rate),
            "large_isi_rate": float(large_isi_rate),
            "waveform_correlation": float(corr),
            "merged_isi_rate": float(merged_isi_rate),
        }

    history_after_student = get_call_history()
    if len(history_after_student) > history_before_student:
        student_model_meta = history_after_student[-1]
    else:
        student_model_meta = {
            "provider": "mock",
            "requested_model": student_model,
            "actual_model": None,
            "usage": {},
        }

    teacher_prompt = build_teacher_feedback_prompt(
        channel=selection.channel,
        step=selection.step,
        action_type=selection.action_type,
        context_summary=teacher_context,
        student_action=student_resp.get("action", ""),
        student_rationale=student_resp.get("rationale", ""),
        human_action=human_action,
        human_reasoning=selection.human_reasoning,
        merge_target_id=selection.target_id,
    )

    history_before_teacher = len(get_call_history())
    teacher_feedback = call_vlm_api(
        prompt=teacher_prompt,
        images=teacher_images,
        model=teacher_model,
        provider=resolved_teacher_provider,
        use_mock=USE_MOCK,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
    )
    history_after_teacher = get_call_history()
    if len(history_after_teacher) > history_before_teacher:
        teacher_model_meta = history_after_teacher[-1]
    else:
        teacher_model_meta = {
            "provider": "mock",
            "requested_model": teacher_model,
            "actual_model": None,
            "usage": {},
        }

    (out_dir / "teacher_prompt.txt").write_text(teacher_prompt)
    (out_dir / "teacher_feedback.txt").write_text(teacher_feedback)

    result = {
        "channel": selection.channel,
        "step": selection.step,
        "run_id": run_id,
        "action_str": selection.action_str,
        "action_type": selection.action_type,
        "cluster_id": selection.cluster_id,
        "target_id": selection.target_id,
        "human_action": human_action,
        "human_reasoning": selection.human_reasoning,
        "student_model": student_model,
        "teacher_model": teacher_model,
        "teacher_provider": resolved_teacher_provider,
        "student_model_actual": student_model_meta.get("actual_model"),
        "teacher_model_actual": teacher_model_meta.get("actual_model"),
        "student_model_meta": student_model_meta,
        "teacher_model_meta": teacher_model_meta,
        "student_action": student_resp.get("action", ""),
        "student_rationale": student_resp.get("rationale", ""),
        "teacher_feedback": teacher_feedback.strip(),
        "output_dir": str(out_dir),
    }

    with open(out_dir / "demo_result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(OUTPUT_BASE / "latest_demo.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one simulated teacher-feedback demo.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--student-model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--teacher-provider", type=str, default=None)
    args = parser.parse_args()

    result = run_demo(
        seed=args.seed,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
    )
    print("\n=== Simulated Teacher Feedback Demo ===")
    print(f"Channel: {result['channel']} | Step: {result['step']} | Action: {result['action_str']}")
    print(f"Models: student={result['student_model']} | teacher={result['teacher_model']}")
    print(f"Student: {result['student_action']} | {result['student_rationale']}")
    print(f"Teacher: {result['teacher_feedback']}")
    print(f"Saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
