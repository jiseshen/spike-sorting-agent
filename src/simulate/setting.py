"""
SettingConfig: per-experiment parameter set for simulated recordings.

Loads from a per-setting YAML, merging over global config.yaml defaults.
CLI overrides take highest priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import yaml


# ---------------------------------------------------------------------------
# Defaults (mirror of config.yaml simulation section)
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "simulation": {
        "n_channels": 10,
        "duration": 120,
        "n_neurons_range": [5, 15],
        "probe": "Neuronexus-32",
        "seed": 0,
    },
    "noise": {
        "level": "medium",
        "noise_level": 15.0,
    },
    "drift": {
        "enabled": False,
        "drift_velocity": 0.0,
    },
    "overlap": {
        "enabled": False,
        "max_overlap_pairs": 0,
    },
    "artifact": {
        "type": "none",
    },
    "teacher_criteria": {
        "min_spike_count": 500,
        "max_isi_violation_rate": 0.006,
        "require_biphasic": True,
        "min_snr": 3.0,
        "teacher_style": "strict",
    },
    "adaptation": {
        "n_train_channels": 8,
        "n_eval_channels": 4,
        "split_seed": 0,
    },
    "sorter": {
        "name": "mountainsort5",
        "detect_threshold": 5.5,
        "snippet_T1": 20,
        "snippet_T2": 20,
    },
    "hierarchy": {
        "enabled": True,
        "similarity_metric": "pearson",
        "min_similarity": 0.9,
        "max_merges": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@dataclass
class SettingConfig:
    """Fully resolved configuration for one experimental setting."""

    setting_id: str

    # --- simulation ---
    n_channels: int = 10
    duration: float = 120.0
    n_neurons_range: Tuple[int, int] = (5, 15)
    probe: str = "Neuronexus-32"
    seed: int = 0

    # --- noise ---
    noise_level: float = 15.0          # uV RMS
    noise_label: str = "medium"        # low | medium | high

    # --- drift ---
    drift_enabled: bool = False
    drift_velocity: float = 0.0        # um/min

    # --- overlap ---
    overlap_enabled: bool = False
    max_overlap_pairs: int = 0

    # --- artifact ---
    artifact_type: str = "none"        # none | periodic | burst

    # --- teacher criteria ---
    min_spike_count: int = 500
    max_isi_violation_rate: float = 0.006
    require_biphasic: bool = True
    min_snr: float = 3.0
    teacher_style: str = "strict"      # strict | liberal | snr_only

    # --- adaptation ---
    n_train_channels: int = 8
    n_eval_channels: int = 4
    split_seed: int = 0

    # --- sorter ---
    sorter_name: str = "mountainsort5"
    detect_threshold: float = 5.5
    snippet_T1: int = 20
    snippet_T2: int = 20

    # --- hierarchy reconstruction (for simulated data) ---
    hierarchy_enabled: bool = True
    hierarchy_similarity_metric: str = "pearson"   # pearson | cosine
    hierarchy_min_similarity: float = 0.9
    hierarchy_max_merges: Optional[int] = None

    @classmethod
    def load(
        cls,
        setting_yaml: str | Path,
        global_config: Optional[str | Path] = None,
        overrides: Optional[dict] = None,
    ) -> "SettingConfig":
        """
        Load a SettingConfig by merging:
          1. Built-in defaults
          2. global_config.yaml (if provided)
          3. per-setting YAML
          4. overrides dict (CLI values)

        Args:
            setting_yaml: Path to configs/settings/<setting_id>.yaml
            global_config: Optional path to root config.yaml
            overrides: Optional flat dict of CLI overrides (e.g. {"n_channels": 5})
        """
        cfg = dict(_DEFAULTS)

        if global_config is not None:
            global_path = Path(global_config)
            if global_path.exists():
                with open(global_path) as f:
                    global_data = yaml.safe_load(f) or {}
                cfg = _deep_merge(cfg, global_data)

        with open(setting_yaml) as f:
            setting_data = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, setting_data)

        if overrides:
            cfg = _deep_merge(cfg, overrides)

        s = cfg.get("simulation", {})
        n = cfg.get("noise", {})
        d = cfg.get("drift", {})
        o = cfg.get("overlap", {})
        a = cfg.get("artifact", {})
        t = cfg.get("teacher_criteria", {})
        ad = cfg.get("adaptation", {})
        so = cfg.get("sorter", {})
        h = cfg.get("hierarchy", {})

        n_neurons_raw = s.get("n_neurons_range", [5, 15])
        n_neurons = tuple(n_neurons_raw) if not isinstance(n_neurons_raw, tuple) else n_neurons_raw

        max_merges_raw = h.get("max_merges", None)
        max_merges = None if max_merges_raw is None else int(max_merges_raw)

        return cls(
            setting_id=cfg.get("setting_id", Path(setting_yaml).stem),
            n_channels=int(s.get("n_channels", 10)),
            duration=float(s.get("duration", 120.0)),
            n_neurons_range=(int(n_neurons[0]), int(n_neurons[1])),
            probe=str(s.get("probe", "Neuronexus-32")),
            seed=int(s.get("seed", 0)),
            noise_level=float(n.get("noise_level", 15.0)),
            noise_label=str(n.get("level", "medium")),
            drift_enabled=bool(d.get("enabled", False)),
            drift_velocity=float(d.get("drift_velocity", 0.0)),
            overlap_enabled=bool(o.get("enabled", False)),
            max_overlap_pairs=int(o.get("max_overlap_pairs", 0)),
            artifact_type=str(a.get("type", "none")),
            min_spike_count=int(t.get("min_spike_count", 500)),
            max_isi_violation_rate=float(t.get("max_isi_violation_rate", 0.006)),
            require_biphasic=bool(t.get("require_biphasic", True)),
            min_snr=float(t.get("min_snr", 3.0)),
            teacher_style=str(t.get("teacher_style", "strict")),
            n_train_channels=int(ad.get("n_train_channels", 8)),
            n_eval_channels=int(ad.get("n_eval_channels", 4)),
            split_seed=int(ad.get("split_seed", 0)),
            sorter_name=str(so.get("name", "mountainsort5")),
            detect_threshold=float(so.get("detect_threshold", 5.5)),
            snippet_T1=int(so.get("snippet_T1", 20)),
            snippet_T2=int(so.get("snippet_T2", 20)),
            hierarchy_enabled=bool(h.get("enabled", True)),
            hierarchy_similarity_metric=str(h.get("similarity_metric", "pearson")),
            hierarchy_min_similarity=float(h.get("min_similarity", 0.9)),
            hierarchy_max_merges=max_merges,
        )

    def to_dict(self) -> dict:
        """Return a plain dict representation suitable for YAML serialization."""
        return {
            "setting_id": self.setting_id,
            "simulation": {
                "n_channels": self.n_channels,
                "duration": self.duration,
                "n_neurons_range": list(self.n_neurons_range),
                "probe": self.probe,
                "seed": self.seed,
            },
            "noise": {"level": self.noise_label, "noise_level": self.noise_level},
            "drift": {"enabled": self.drift_enabled, "drift_velocity": self.drift_velocity},
            "overlap": {"enabled": self.overlap_enabled, "max_overlap_pairs": self.max_overlap_pairs},
            "artifact": {"type": self.artifact_type},
            "teacher_criteria": {
                "min_spike_count": self.min_spike_count,
                "max_isi_violation_rate": self.max_isi_violation_rate,
                "require_biphasic": self.require_biphasic,
                "min_snr": self.min_snr,
                "teacher_style": self.teacher_style,
            },
            "adaptation": {
                "n_train_channels": self.n_train_channels,
                "n_eval_channels": self.n_eval_channels,
                "split_seed": self.split_seed,
            },
            "sorter": {
                "name": self.sorter_name,
                "detect_threshold": self.detect_threshold,
                "snippet_T1": self.snippet_T1,
                "snippet_T2": self.snippet_T2,
            },
            "hierarchy": {
                "enabled": self.hierarchy_enabled,
                "similarity_metric": self.hierarchy_similarity_metric,
                "min_similarity": self.hierarchy_min_similarity,
                "max_merges": self.hierarchy_max_merges,
            },
        }
