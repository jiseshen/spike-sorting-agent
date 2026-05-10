"""
MEArec-based recording generator.

Produces one independent recording per channel index under a setting, saves:
  output/<setting_id>/<channel_id>/raw/recording.h5
  output/<setting_id>/<channel_id>/raw/metadata.json
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import inspect
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np

from .setting import SettingConfig


def _compiled_mechanism_exists(mods_dir: Path) -> bool:
    """Check whether NEURON compiled mechanism artifacts exist under mods/."""
    if not mods_dir.exists():
        return False
    return any(mods_dir.rglob("libnrnmech*")) or any(mods_dir.rglob("special"))


def _ensure_nrnivmodl_available_in_path() -> None:
    """Ensure nrnivmodl from the active Python environment is available in PATH."""
    py_bin = Path(sys.executable).resolve().parent
    nrnivmodl = py_bin / "nrnivmodl"
    if nrnivmodl.exists():
        path = os.environ.get("PATH", "")
        if str(py_bin) not in path.split(":"):
            os.environ["PATH"] = f"{py_bin}:{path}" if path else str(py_bin)


def _compile_mechanisms_if_needed(mr_module, cell_models_folder: Path) -> None:
    """
    Compile MEArec/NEURON mechanisms if compiled artifacts are missing.

    This defends against environment mismatches where nrnivmodl exists in the
    venv but is not visible in PATH when MEArec invokes compile helpers.
    """
    mods_dir = cell_models_folder / "mods"
    if _compiled_mechanism_exists(mods_dir):
        return

    _ensure_nrnivmodl_available_in_path()
    simulate_cells_py = Path(mr_module.__file__).resolve().parent / "simulate_cells.py"
    cmd = [sys.executable, str(simulate_cells_py), "compile", str(cell_models_folder)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or (not _compiled_mechanism_exists(mods_dir)):
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            "Failed to compile NEURON mechanisms for MEArec.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout (tail): {stdout[-1000:]}\n"
            f"stderr (tail): {stderr[-1000:]}"
        )


def _assert_neuron_version_compatible() -> None:
    """
    Fail fast for known-incompatible NEURON versions.

    MEArec bundled BBP mechanism files still rely on legacy random API calls
    that fail to compile with NEURON >= 9 in many environments.
    """
    try:
        v = metadata.version("neuron")
    except Exception:
        return

    major_str = v.split(".")[0]
    try:
        major = int(major_str)
    except ValueError:
        return

    if major >= 9:
        raise RuntimeError(
            f"Detected neuron=={v}, which is often incompatible with MEArec BBP MOD files. "
            "Please use neuron<9 (e.g. `python3 -m pip install \"neuron<9\"`) "
            "and recompile mechanisms with `nrnivmodl` in the MEArec mods folder."
        )


def _to_builtin_scalars(obj):
    """
    Recursively convert numpy scalar types to native Python scalar types.

    MEArec 1.9.x HDF5 saver rejects some numpy scalar subclasses
    (notably np.float32) in nested annotation dicts.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_builtin_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin_scalars(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_builtin_scalars(v) for v in obj)
    return obj


def _sanitize_spiketrain_annotations_for_save(recgen) -> None:
    """Normalize annotation scalar types so MEArec save_recording_generator succeeds."""
    spiketrains = getattr(recgen, "spiketrains", None)
    if spiketrains is None:
        return
    for st in spiketrains:
        ann = getattr(st, "annotations", None)
        if isinstance(ann, dict):
            st.annotations = _to_builtin_scalars(ann)


def generate_recording(
    cfg: SettingConfig,
    channel_idx: int,
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    """
    Generate one MEArec recording for (setting, channel_idx).

    Args:
        cfg: Resolved SettingConfig for this setting.
        channel_idx: Zero-based channel index within the setting (0 … n_channels-1).
        output_dir: Root output directory (e.g. "output/").
        force: Re-generate even if output already exists.

    Returns:
        Path to the channel raw/ directory.
    """
    try:
        import MEArec as mr
    except ImportError as e:
        raise ImportError(
            "MEArec is required for data simulation. "
            "Install with: pip install MEArec"
        ) from e

    channel_id = f"ch_{channel_idx:03d}"
    raw_dir = Path(output_dir) / cfg.setting_id / channel_id / "raw"

    if raw_dir.exists() and (raw_dir / "recording.h5").exists() and not force:
        print(f"  [skip] {raw_dir} already exists (use --force to overwrite)")
        return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Per-channel seed derived from setting seed + channel index for reproducibility
    channel_seed = cfg.seed + channel_idx
    rng = random.Random(channel_seed)
    n_neurons = rng.randint(cfg.n_neurons_range[0], cfg.n_neurons_range[1])
    # Keep a larger template pool than final neuron count so MEArec selection
    # can satisfy min_dist / amplitude constraints robustly.
    n_templates_pool = max(80, n_neurons * 12)

    # --- MEArec template generation ---
    try:
        cell_models_folder = mr.get_default_cell_models_folder()
    except Exception:
        cell_models_folder = None

    if cell_models_folder is None:
        cfg_info, _ = mr.get_default_config()
        cell_models_folder = cfg_info.get("cell_models_folder")

    if cell_models_folder is None or (not Path(cell_models_folder).exists()):
        raise FileNotFoundError(
            "MEArec cell models folder not found. "
            "Please install/download MEArec cell models, then retry. "
            f"Resolved path: {cell_models_folder}"
        )
    _assert_neuron_version_compatible()
    cell_models_folder = Path(cell_models_folder).resolve()
    _compile_mechanisms_if_needed(mr, cell_models_folder)

    tetrode_name = cfg.probe
    tempgen = mr.gen_templates(
        cell_models_folder=str(cell_models_folder),
        params={
            "probe": tetrode_name,
            "n": n_templates_pool,
            "drifting": bool(cfg.drift_enabled),
            "seed": channel_seed,
        },
    )

    # --- Firing activity (Poisson) ---
    spgen = mr.gen_spiketrains(
        params={
            "n_exc": n_neurons,
            "n_inh": 0,
            "duration": cfg.duration,
            "seed": channel_seed,
        },
        seed=channel_seed,
    )

    # --- Drift params ---
    drift_dict: dict = {}
    if cfg.drift_enabled:
        # MEArec drift expects drift mode + explicit slow/fast parameters.
        drift_dict = {
            "drifting": True,
            "drift_mode_speed": "slow",
            "drift_mode_probe": "rigid",
            "slow_drift_velocity": max(float(cfg.drift_velocity), 0.1),
        }

    # --- Recording generation ---
    recording_params = {
        "recordings": {
            "noise_level": cfg.noise_level,
            "overlap": bool(cfg.overlap_enabled),
            **drift_dict,
        },
        "templates": {
            "n_overlap_pairs": cfg.max_overlap_pairs if cfg.max_overlap_pairs > 0 else None,
            # Relax defaults to reduce template-selection failures in noisy settings.
            "min_dist": 10,
            "min_amp": 30,
            "max_amp": 500,
        },
        "seeds": {
            "spiketrains": channel_seed,
            "templates": channel_seed,
            "convolution": channel_seed,
            "noise": channel_seed,
        },
    }

    sig = inspect.signature(mr.gen_recordings)
    if "spgen" in sig.parameters:
        recgen = mr.gen_recordings(
            tempgen=tempgen,
            spgen=spgen,
            params=recording_params,
        )
    else:
        # Backward-compatible fallback for older API variants.
        recgen = mr.gen_recordings(
            templates=tempgen,
            spiketrains=spgen,
            params=recording_params,
        )

    rec_path = raw_dir / "recording.h5"
    _sanitize_spiketrain_annotations_for_save(recgen)
    mr.save_recording_generator(recgen, str(rec_path))

    metadata = {
        "setting_id": cfg.setting_id,
        "channel_id": channel_id,
        "channel_idx": channel_idx,
        "n_neurons": n_neurons,
        "duration": cfg.duration,
        "noise_level": cfg.noise_level,
        "noise_label": cfg.noise_label,
        "drift_enabled": cfg.drift_enabled,
        "drift_velocity": cfg.drift_velocity,
        "overlap_enabled": cfg.overlap_enabled,
        "artifact_type": cfg.artifact_type,
        "probe": cfg.probe,
        "seed": channel_seed,
    }
    with open(raw_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [done] Generated {channel_id}: {n_neurons} neurons → {rec_path}")
    return raw_dir
