"""
ABLATION TEST: VLM runner WITHOUT quantitative metrics in prompts.

This module is identical to vlm_runner.py except:
1. Uses agent_context_ablation for prompts (no numerical metrics)
2. Does NOT include spike counts, ISI rates, correlations in prompts
3. VLM makes decisions based ONLY on visual information

All image generation and API call logic remains the same.
"""

import re
import json
import time
import base64
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from matplotlib.figure import Figure

from .agent_context_ablation import (
    NEURONAL_CRITERIA,
    build_phase1_prompt_ablation,
    build_phase2_prompt_ablation,
)

# Import VLM API
try:
    from .vlm_api import call_vlm
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    print("[Warning] VLM API not available, using mock responses")


# =====================================================================
# IMAGE UTILITIES - Reuse from original vlm_runner.py
# =====================================================================

_VLM_CALL_INDEX = 0  # Monotonic counter for unique logging filenames per process

def _save_vlm_inputs(
    output_dir: Optional[Path],
    prefix: str,
    images: List[str],
    prompt: str,
    image_names: List[str],
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """Save VLM input images and prompt to output directory with a UNIQUE call suffix."""
    if output_dir is None:
        return

    global _VLM_CALL_INDEX
    _VLM_CALL_INDEX += 1
    call_id = _VLM_CALL_INDEX
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_prefix = f"{prefix}_call{call_id:05d}"

    # Create vlm_inputs subdirectory
    vlm_dir = output_dir / "vlm_inputs"
    vlm_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt
    prompt_file = vlm_dir / f"{unique_prefix}_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)

    saved_image_files: List[str] = []
    # Save images
    for img_b64, img_name in zip(images, image_names):
        img_bytes = base64.b64decode(img_b64)
        img_file = vlm_dir / f"{unique_prefix}_{img_name}.png"
        with open(img_file, 'wb') as f:
            f.write(img_bytes)
        saved_image_files.append(img_file.name)

    # Append to CSV log (create header if not exists)
    log_path = vlm_dir / "vlm_call_log.csv"
    header_needed = not log_path.exists()
    row = {
        "call_id": call_id,
        "timestamp": timestamp,
        "base_prefix": prefix,
        "unique_prefix": unique_prefix,
        "prompt_file": prompt_file.name,
        "image_files": ";".join(saved_image_files),
    }
    if extra_meta:
        for k, v in extra_meta.items():
            row[k] = v

    # Ensure consistent column ordering when appending
    import csv
    if header_needed:
        with open(log_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
    else:
        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            writer.writerow(row)
        


def _sanitize_json_response(text: str) -> str:
    """Sanitize model output into strict JSON."""
    import re

    # Strip code fences
    text = re.sub(r"```(?:json)?\n?|```", "", text, flags=re.IGNORECASE)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Find the first '{' and extract a balanced JSON object
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")

    # Scan to find the matching closing brace accounting for braces inside strings
    brace_count = 0
    in_string = False
    escape = False
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
    if end is None:
        raise ValueError("No matching closing brace found")

    json_str = text[start:end+1]

    # Remove // comments
    json_str = re.sub(r"//.*", "", json_str)

    # Remove trailing commas before ] or }
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)

    return json_str

def fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


# Import all image creation utilities from original vlm_runner
from .vlm_runner import (
    create_waveform_overlay_image,
    create_isi_histogram_image,
    create_amplitude_distribution_image,
    create_aggregation_tree_image,
    create_waveform_comparison_image,
    compute_waveform_correlation,
    compute_merged_isi_violation_rate,
)


# =====================================================================
# VLM API INTERFACE
# =====================================================================

def call_vlm_api(
    prompt: str,
    images: List[str],
    model: str = "gpt-4o",
    provider: str = "gpt4o",
    use_mock: bool = False,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Call VLM API with retry logic."""
    if use_mock:
        return '{"action": "KEEP", "rationale": "Mock response"}'
    
    if not VLM_AVAILABLE:
        raise RuntimeError("VLM API not available and use_mock=False")
    
    for attempt in range(max_retries):
        try:
            response = call_vlm(
                prompt=prompt,
                images=images,
                model=model,
                provider=provider,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[VLM API Error] Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"[VLM API Error] All {max_retries} attempts failed")
                raise


# =====================================================================
# PHASE 1: VLM ITERATIVE SPLIT/DISCARD (ABLATION VERSION)
# =====================================================================

def vlm_phase1_cluster_decision(
    cluster_id: int,
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    overcluster_composition: List[int],
    hierarchy_tree: np.ndarray,
    sampling_rate: float = 30000.0,
    provider: str = "gpt4o",
    model: str = "gpt-4o",
    use_mock: bool = False,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    ABLATION: Phase 1 decision WITHOUT quality metrics in prompt.
    
    Identical to original except:
    - Uses build_phase1_prompt_ablation() which excludes ISI rates, correlations, CV values
    - Still includes spike counts and cluster size info
    - VLM decides based on images + structural information
    
    Returns:
        Dict with:
            - action: "KEEP" | "DISCARD" | "SPLIT"
            - rationale: Reasoning string
            - raw_response: Original VLM response
    """
    # Create visualizations (same as original)
    wf_overlay = create_waveform_overlay_image(waveforms, cluster_id, sampling_rate)
    isi_hist = create_isi_histogram_image(spike_times, cluster_id)
    agg_tree = create_aggregation_tree_image(hierarchy_tree, overcluster_composition, cluster_id)
    
    # Build ABLATION prompt (includes spike counts, excludes quality metrics)
    n_spikes = len(spike_times)
    n_overclusters = len(overcluster_composition)
    prompt = build_phase1_prompt_ablation(
        cluster_id=cluster_id,
        n_spikes=n_spikes,
        n_overclusters=n_overclusters,
    )
    
    images = [wf_overlay, isi_hist, agg_tree]
    image_names = ["waveform_overlay", "isi_histogram", "aggregation_tree"]
    
    # Save inputs if output_dir provided
    if output_dir is not None:
        _save_vlm_inputs(
            output_dir=output_dir,
            prefix=f"phase1_cluster_{cluster_id}",
            images=images,
            prompt=prompt,
            image_names=image_names,
            extra_meta={"phase": "1", "cluster_id": cluster_id, "n_spikes": len(spike_times)},
        )
    
    # Call VLM with retry on parse failure (max 3 attempts)
    max_parse_attempts = 3
    for parse_attempt in range(max_parse_attempts):
        raw_response = call_vlm_api(
            prompt=prompt,
            images=images,
            model=model,
            provider=provider,
            use_mock=use_mock,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        
        # Parse JSON (robust to code fences/comments/trailing commas)
        try:
            sanitized = _sanitize_json_response(raw_response)
            decision = json.loads(sanitized)
            action = decision.get("action", "DISCARD")
            if action not in ["KEEP", "DISCARD", "SPLIT"]:
                action = "DISCARD"
            return {
                "action": action,
                "rationale": decision.get("rationale", ""),
                "raw_response": raw_response,
            }
        except Exception as e:
            print(f"[VLM Parse Error] Attempt {parse_attempt + 1}/{max_parse_attempts}: {e}")
            print(f"[VLM Raw Response] {raw_response[:500]}...")
            
            if parse_attempt < max_parse_attempts - 1:
                print(f"[VLM] Retrying VLM call due to parse failure...")
                time.sleep(1)  # Brief pause before retry
            else:
                # All retries exhausted, fall back to default
                print(f"[VLM] All {max_parse_attempts} parse attempts failed, falling back to DISCARD")
                return {
                    "action": "DISCARD",
                    "rationale": f"JSON parse error after {max_parse_attempts} attempts",
                    "raw_response": raw_response,
                }


# =====================================================================
# PHASE 2: VLM MERGE/DISCARD DECISIONS (ABLATION VERSION)
# =====================================================================

def vlm_phase2_merge_decision(
    small_cluster_id: int,
    small_waveforms: np.ndarray,
    small_spike_times: np.ndarray,
    large_cluster_id: int,
    large_waveforms: np.ndarray,
    large_spike_times: np.ndarray,
    sampling_rate: float = 30000.0,
    provider: str = "gpt4o",
    model: str = "gpt-4o",
    use_mock: bool = False,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    ABLATION: Phase 2 merge decision WITHOUT quality metrics in prompt.
    
    Identical to original except:
    - Uses build_phase2_prompt_ablation() which excludes ISI rates and correlations
    - Still includes spike counts and total after merge
    - VLM decides based on images + size information
    
    Returns:
        Dict with:
            - action: "MERGE" | "NOT_MERGE" | "DISCARD"
            - rationale: Reasoning string
            - raw_response: Original VLM response
    """
    # Create visualizations (same as original)
    small_wf_img, large_wf_img, merged_isi_img = create_waveform_comparison_image(
        small_waveforms=small_waveforms,
        large_waveforms=large_waveforms,
        small_spike_times=small_spike_times,
        large_spike_times=large_spike_times,
        small_cluster_id=small_cluster_id,
        large_cluster_id=large_cluster_id,
        sampling_rate=sampling_rate,
    )
    
    # Build ABLATION prompt (includes spike counts, excludes quality metrics)
    n_small = len(small_spike_times)
    n_large = len(large_spike_times)
    prompt = build_phase2_prompt_ablation(
        small_cluster_id=small_cluster_id,
        n_small=n_small,
        large_cluster_id=large_cluster_id,
        n_large=n_large,
    )
    
    images = [small_wf_img, large_wf_img, merged_isi_img]
    image_names = [
        f"small_cluster_{small_cluster_id}_waveforms",
        f"large_cluster_{large_cluster_id}_waveforms",
        "merged_isi_histogram",
    ]
    
    # Save inputs if output_dir provided
    if output_dir is not None:
        _save_vlm_inputs(
            output_dir=output_dir,
            prefix=f"phase2_merge_{small_cluster_id}_to_{large_cluster_id}",
            images=images,
            prompt=prompt,
            image_names=image_names,
            extra_meta={
                "phase": "2",
                "small_cluster_id": small_cluster_id,
                "large_cluster_id": large_cluster_id,
                "n_small": len(small_spike_times),
                "n_large": len(large_spike_times),
            },
        )
    
    # Call VLM with retry on parse failure (max 3 attempts)
    max_parse_attempts = 3
    for parse_attempt in range(max_parse_attempts):
        raw_response = call_vlm_api(
            prompt=prompt,
            images=images,
            model=model,
            provider=provider,
            use_mock=use_mock,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        
        # Parse JSON (robust to code fences/comments/trailing commas)
        try:
            sanitized = _sanitize_json_response(raw_response)
            decision = json.loads(sanitized)
            action = decision.get("action", "DISCARD")
            if action not in ["MERGE", "NOT_MERGE", "DISCARD"]:
                action = "NOT_MERGE"  # Default to NOT_MERGE for invalid actions
            return {
                "action": action,
                "rationale": decision.get("rationale", ""),
                "raw_response": raw_response,
            }
        except Exception as e:
            print(f"[VLM Parse Error] Attempt {parse_attempt + 1}/{max_parse_attempts}: {e}")
            print(f"[VLM Raw Response] {raw_response[:500]}...")
            
            if parse_attempt < max_parse_attempts - 1:
                print(f"[VLM] Retrying VLM call due to parse failure...")
                time.sleep(1)  # Brief pause before retry
            else:
                # All retries exhausted, fall back to default
                print(f"[VLM] All {max_parse_attempts} parse attempts failed, falling back to NOT_MERGE")
                return {
                    "action": "NOT_MERGE",
                    "rationale": f"JSON parse error after {max_parse_attempts} attempts",
                    "raw_response": raw_response,
                }
