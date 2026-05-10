"""Continual RAG memory for spike-sorting decision support.

This module stores structured evidence (raw waveform-derived vectors, numeric
features, GT action/reasoning, and prompt/image traces) and retrieves similar
past cases for few-shot prompting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_float_list(arr: np.ndarray) -> List[float]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return [float(x) for x in flat.tolist()]


def _safe_cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    n = min(va.size, vb.size)
    if n == 0:
        return 0.0
    va = va[:n]
    vb = vb[:n]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def compute_waveform_template(waveforms: np.ndarray) -> List[float]:
    wf = np.asarray(waveforms)
    if wf.size == 0:
        return []
    if wf.ndim == 1:
        return _to_float_list(wf)
    return _to_float_list(np.median(wf, axis=0))


def sample_waveforms(waveforms: np.ndarray, max_waveforms: int = 64) -> List[List[float]]:
    wf = np.asarray(waveforms)
    if wf.size == 0:
        return []
    if wf.ndim == 1:
        return [
            _to_float_list(wf),
        ]
    n = wf.shape[0]
    if n <= max_waveforms:
        chosen = wf
    else:
        # Deterministic downsample for stable memory snapshots.
        idx = np.linspace(0, n - 1, num=max_waveforms, dtype=np.int64)
        chosen = wf[idx]
    return [_to_float_list(row) for row in chosen]


def compute_isi_rate(spike_times: np.ndarray, refractory_period_ms: float = 2.0) -> float:
    st = np.asarray(spike_times, dtype=np.float64).reshape(-1)
    if st.size < 2:
        return 0.0
    dt = np.diff(np.sort(st))
    if dt.size == 0:
        return 0.0
    threshold = refractory_period_ms / 1000.0
    return float(np.mean(dt < threshold))


def compute_amplitude_stats(waveforms: np.ndarray) -> Dict[str, float]:
    wf = np.asarray(waveforms)
    if wf.size == 0:
        return {
            "amplitude_mean": 0.0,
            "amplitude_std": 0.0,
            "amplitude_cv": 0.0,
            "peak_to_trough_mean": 0.0,
        }

    if wf.ndim == 1:
        peak_to_trough = np.ptp(wf)
        return {
            "amplitude_mean": float(peak_to_trough),
            "amplitude_std": 0.0,
            "amplitude_cv": 0.0,
            "peak_to_trough_mean": float(peak_to_trough),
        }

    ptp = np.ptp(wf, axis=1)
    amp_mean = float(np.mean(ptp)) if ptp.size else 0.0
    amp_std = float(np.std(ptp, ddof=1)) if ptp.size > 1 else 0.0
    amp_cv = float(amp_std / amp_mean) if amp_mean > 1e-12 else 0.0
    return {
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "amplitude_cv": amp_cv,
        "peak_to_trough_mean": amp_mean,
    }


def compute_waveform_correlation(
    waveforms_a: np.ndarray,
    waveforms_b: np.ndarray,
    max_samples: int = 200,
) -> float:
    wa = np.asarray(waveforms_a)
    wb = np.asarray(waveforms_b)
    if wa.size == 0 or wb.size == 0:
        return 0.0

    if wa.ndim == 2 and wa.shape[0] > max_samples:
        idx = np.linspace(0, wa.shape[0] - 1, num=max_samples, dtype=np.int64)
        wa = wa[idx]
    if wb.ndim == 2 and wb.shape[0] > max_samples:
        idx = np.linspace(0, wb.shape[0] - 1, num=max_samples, dtype=np.int64)
        wb = wb[idx]

    ta = np.median(wa, axis=0) if wa.ndim == 2 else wa.reshape(-1)
    tb = np.median(wb, axis=0) if wb.ndim == 2 else wb.reshape(-1)
    if ta.size == 0 or tb.size == 0:
        return 0.0

    n = min(ta.size, tb.size)
    ta = ta[:n].astype(np.float32)
    tb = tb[:n].astype(np.float32)
    ta = ta - np.mean(ta)
    tb = tb - np.mean(tb)
    denom = float(np.linalg.norm(ta) * np.linalg.norm(tb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(ta, tb) / denom)


def compute_merged_isi_rate(
    spike_times_a: np.ndarray,
    spike_times_b: np.ndarray,
    refractory_period_ms: float = 2.0,
) -> float:
    a = np.asarray(spike_times_a, dtype=np.float64).reshape(-1)
    b = np.asarray(spike_times_b, dtype=np.float64).reshape(-1)
    if a.size == 0 and b.size == 0:
        return 0.0
    merged = np.sort(np.concatenate([a, b]))
    return compute_isi_rate(merged, refractory_period_ms=refractory_period_ms)


def build_phase1_feature_vector(
    n_spikes: int,
    isi_rate: float,
    amplitude_stats: Dict[str, float],
) -> List[float]:
    return [
        float(np.log1p(max(0, n_spikes)) / 10.0),
        float(isi_rate),
        float(np.log1p(max(0.0, amplitude_stats.get("amplitude_mean", 0.0)))),
        float(np.log1p(max(0.0, amplitude_stats.get("amplitude_std", 0.0)))),
        float(amplitude_stats.get("amplitude_cv", 0.0)),
        float(np.log1p(max(0.0, amplitude_stats.get("peak_to_trough_mean", 0.0)))),
    ]


def build_phase2_feature_vector(
    n_small: int,
    n_large: int,
    small_isi: float,
    large_isi: float,
    merged_isi: float,
    waveform_corr: float,
    small_amp_stats: Dict[str, float],
    large_amp_stats: Dict[str, float],
) -> List[float]:
    small_amp_mean_log = float(np.log1p(max(0.0, small_amp_stats.get("amplitude_mean", 0.0))))
    large_amp_mean_log = float(np.log1p(max(0.0, large_amp_stats.get("amplitude_mean", 0.0))))
    return [
        float(np.log1p(max(0, n_small)) / 10.0),
        float(np.log1p(max(0, n_large)) / 10.0),
        float(small_isi),
        float(large_isi),
        float(merged_isi),
        float(waveform_corr),
        float(small_amp_stats.get("amplitude_cv", 0.0)),
        float(large_amp_stats.get("amplitude_cv", 0.0)),
        abs(small_amp_mean_log - large_amp_mean_log),
    ]


@dataclass
class MemoryEntry:
    channel_id: str
    step: int
    phase: str
    cluster_id: int
    target_id: Optional[int]

    waveform_template: List[float]
    waveform_sample: List[List[float]]

    n_spikes: int
    isi_rate: float
    amplitude_stats: Dict[str, float]
    waveform_corr: Optional[float] = None
    merged_isi_rate: Optional[float] = None

    gt_action: Optional[str] = None
    gt_reasoning: Optional[str] = None

    prompt_text: str = ""
    image_paths: List[str] = field(default_factory=list)

    feature_vector: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            channel_id=str(payload.get("channel_id", "")),
            step=int(payload.get("step", 0)),
            phase=str(payload.get("phase", "")),
            cluster_id=int(payload.get("cluster_id", 0)),
            target_id=(
                None
                if payload.get("target_id") is None
                else int(payload.get("target_id"))
            ),
            waveform_template=[float(x) for x in payload.get("waveform_template", [])],
            waveform_sample=[
                [float(v) for v in row] for row in payload.get("waveform_sample", [])
            ],
            n_spikes=int(payload.get("n_spikes", 0)),
            isi_rate=_safe_float(payload.get("isi_rate", 0.0)),
            amplitude_stats={
                "amplitude_mean": _safe_float(payload.get("amplitude_stats", {}).get("amplitude_mean", 0.0)),
                "amplitude_std": _safe_float(payload.get("amplitude_stats", {}).get("amplitude_std", 0.0)),
                "amplitude_cv": _safe_float(payload.get("amplitude_stats", {}).get("amplitude_cv", 0.0)),
                "peak_to_trough_mean": _safe_float(payload.get("amplitude_stats", {}).get("peak_to_trough_mean", 0.0)),
            },
            waveform_corr=(
                None
                if payload.get("waveform_corr") is None
                else _safe_float(payload.get("waveform_corr"))
            ),
            merged_isi_rate=(
                None
                if payload.get("merged_isi_rate") is None
                else _safe_float(payload.get("merged_isi_rate"))
            ),
            gt_action=(None if payload.get("gt_action") is None else str(payload.get("gt_action"))),
            gt_reasoning=(
                None if payload.get("gt_reasoning") is None else str(payload.get("gt_reasoning"))
            ),
            prompt_text=str(payload.get("prompt_text", "")),
            image_paths=[str(p) for p in payload.get("image_paths", [])],
            feature_vector=[float(x) for x in payload.get("feature_vector", [])],
        )


class ContinualRAGMemory:
    """Append-only structured memory with similarity retrieval."""

    def __init__(
        self,
        memory_path: Optional[str | Path] = None,
        waveform_weight: float = 0.7,
        feature_weight: float = 0.3,
        default_top_k: int = 3,
    ) -> None:
        self.memory_path: Optional[Path] = Path(memory_path) if memory_path else None
        self.waveform_weight = float(waveform_weight)
        self.feature_weight = float(feature_weight)
        self.default_top_k = int(default_top_k)
        self.entries: List[MemoryEntry] = []
        if self.memory_path is not None and self.memory_path.exists():
            self.load()

    def clear(self, persist: bool = True) -> None:
        self.entries = []
        if persist:
            self.save()

    def load(self) -> None:
        if self.memory_path is None or not self.memory_path.exists():
            return
        loaded: List[MemoryEntry] = []
        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                loaded.append(MemoryEntry.from_dict(json.loads(line)))
        self.entries = loaded

    def save(self, path: Optional[str | Path] = None) -> None:
        target = Path(path) if path else self.memory_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        if self.memory_path is None:
            return
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def retrieve(
        self,
        *,
        query_waveform_template: List[float],
        query_feature_vector: List[float],
        phase: Optional[str] = None,
        top_k: Optional[int] = None,
        require_gt: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.entries:
            return []

        k = self.default_top_k if top_k is None else max(0, int(top_k))
        if k == 0:
            return []

        ranked: List[Dict[str, Any]] = []
        for entry in self.entries:
            if phase is not None and entry.phase != phase:
                continue
            if require_gt and (not entry.gt_action or not (entry.gt_reasoning or "").strip()):
                continue

            sim_w = _safe_cosine(query_waveform_template, entry.waveform_template)
            sim_f = _safe_cosine(query_feature_vector, entry.feature_vector)
            score = self.waveform_weight * sim_w + self.feature_weight * sim_f

            item = entry.to_dict()
            item["score"] = float(score)
            item["waveform_similarity"] = float(sim_w)
            item["feature_similarity"] = float(sim_f)
            ranked.append(item)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:k]

    def retrieve_phase1(
        self,
        *,
        waveforms: np.ndarray,
        spike_times: np.ndarray,
        n_spikes: int,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        amp = compute_amplitude_stats(waveforms)
        query_template = compute_waveform_template(waveforms)
        query_feature = build_phase1_feature_vector(
            n_spikes=n_spikes,
            isi_rate=compute_isi_rate(spike_times),
            amplitude_stats=amp,
        )
        return self.retrieve(
            query_waveform_template=query_template,
            query_feature_vector=query_feature,
            phase="phase1",
            top_k=top_k,
            require_gt=True,
        )

    def retrieve_phase2(
        self,
        *,
        small_waveforms: np.ndarray,
        small_spike_times: np.ndarray,
        large_waveforms: np.ndarray,
        large_spike_times: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        small_template = compute_waveform_template(small_waveforms)
        large_template = compute_waveform_template(large_waveforms)
        query_template = small_template + large_template

        small_amp = compute_amplitude_stats(small_waveforms)
        large_amp = compute_amplitude_stats(large_waveforms)
        query_feature = build_phase2_feature_vector(
            n_small=int(np.asarray(small_spike_times).reshape(-1).size),
            n_large=int(np.asarray(large_spike_times).reshape(-1).size),
            small_isi=compute_isi_rate(small_spike_times),
            large_isi=compute_isi_rate(large_spike_times),
            merged_isi=compute_merged_isi_rate(small_spike_times, large_spike_times),
            waveform_corr=compute_waveform_correlation(small_waveforms, large_waveforms),
            small_amp_stats=small_amp,
            large_amp_stats=large_amp,
        )
        return self.retrieve(
            query_waveform_template=query_template,
            query_feature_vector=query_feature,
            phase="phase2",
            top_k=top_k,
            require_gt=True,
        )


def build_phase1_memory_entry(
    *,
    channel_id: str,
    step: int,
    cluster_id: int,
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    gt_action: Optional[str],
    gt_reasoning: Optional[str],
    prompt_text: str,
    image_paths: List[str],
    target_id: Optional[int] = None,
    max_waveform_sample: int = 64,
) -> MemoryEntry:
    amp = compute_amplitude_stats(waveforms)
    n_spikes = int(np.asarray(spike_times).reshape(-1).size)
    isi_rate = compute_isi_rate(spike_times)
    return MemoryEntry(
        channel_id=channel_id,
        step=int(step),
        phase="phase1",
        cluster_id=int(cluster_id),
        target_id=target_id,
        waveform_template=compute_waveform_template(waveforms),
        waveform_sample=sample_waveforms(waveforms, max_waveforms=max_waveform_sample),
        n_spikes=n_spikes,
        isi_rate=isi_rate,
        amplitude_stats=amp,
        waveform_corr=None,
        merged_isi_rate=None,
        gt_action=gt_action,
        gt_reasoning=gt_reasoning,
        prompt_text=prompt_text,
        image_paths=[str(p) for p in image_paths],
        feature_vector=build_phase1_feature_vector(
            n_spikes=n_spikes,
            isi_rate=isi_rate,
            amplitude_stats=amp,
        ),
    )


def build_phase2_memory_entry(
    *,
    channel_id: str,
    step: int,
    cluster_id: int,
    target_id: int,
    small_waveforms: np.ndarray,
    small_spike_times: np.ndarray,
    large_waveforms: np.ndarray,
    large_spike_times: np.ndarray,
    gt_action: Optional[str],
    gt_reasoning: Optional[str],
    prompt_text: str,
    image_paths: List[str],
    max_waveform_sample: int = 32,
) -> MemoryEntry:
    small_amp = compute_amplitude_stats(small_waveforms)
    large_amp = compute_amplitude_stats(large_waveforms)
    n_small = int(np.asarray(small_spike_times).reshape(-1).size)
    n_large = int(np.asarray(large_spike_times).reshape(-1).size)

    small_template = compute_waveform_template(small_waveforms)
    large_template = compute_waveform_template(large_waveforms)
    waveform_corr = compute_waveform_correlation(small_waveforms, large_waveforms)
    merged_isi = compute_merged_isi_rate(small_spike_times, large_spike_times)

    amplitude_stats = {
        "amplitude_mean": small_amp["amplitude_mean"],
        "amplitude_std": small_amp["amplitude_std"],
        "amplitude_cv": small_amp["amplitude_cv"],
        "peak_to_trough_mean": small_amp["peak_to_trough_mean"],
        "small_amplitude_mean": small_amp["amplitude_mean"],
        "large_amplitude_mean": large_amp["amplitude_mean"],
        "small_amplitude_cv": small_amp["amplitude_cv"],
        "large_amplitude_cv": large_amp["amplitude_cv"],
    }

    waveform_sample_combined = sample_waveforms(
        small_waveforms, max_waveforms=max_waveform_sample
    ) + sample_waveforms(large_waveforms, max_waveforms=max_waveform_sample)

    return MemoryEntry(
        channel_id=channel_id,
        step=int(step),
        phase="phase2",
        cluster_id=int(cluster_id),
        target_id=int(target_id),
        waveform_template=small_template + large_template,
        waveform_sample=waveform_sample_combined,
        n_spikes=n_small,
        isi_rate=compute_isi_rate(small_spike_times),
        amplitude_stats=amplitude_stats,
        waveform_corr=waveform_corr,
        merged_isi_rate=merged_isi,
        gt_action=gt_action,
        gt_reasoning=gt_reasoning,
        prompt_text=prompt_text,
        image_paths=[str(p) for p in image_paths],
        feature_vector=build_phase2_feature_vector(
            n_small=n_small,
            n_large=n_large,
            small_isi=compute_isi_rate(small_spike_times),
            large_isi=compute_isi_rate(large_spike_times),
            merged_isi=merged_isi,
            waveform_corr=waveform_corr,
            small_amp_stats=small_amp,
            large_amp_stats=large_amp,
        ),
    )


def format_retrieved_examples_for_prompt(
    retrieved_examples: List[Dict[str, Any]],
    max_reason_chars: int = 220,
) -> str:
    if not retrieved_examples:
        return ""

    lines: List[str] = []
    lines.append("## Few-shot from Similar Past Cases (historical only)")
    lines.append("Use these as references; do not copy wording verbatim.")
    for idx, ex in enumerate(retrieved_examples, start=1):
        gt_action = ex.get("gt_action") or "UNKNOWN"
        reason = (ex.get("gt_reasoning") or "").strip()
        if len(reason) > max_reason_chars:
            reason = reason[: max_reason_chars - 3] + "..."
        lines.append(
            (
                f"Case {idx} | phase={ex.get('phase')} | score={ex.get('score', 0.0):.3f} "
                f"(wf={ex.get('waveform_similarity', 0.0):.3f}, feat={ex.get('feature_similarity', 0.0):.3f})"
            )
        )
        lines.append(
            f"- cluster={ex.get('cluster_id')} target={ex.get('target_id')} n_spikes={ex.get('n_spikes')} isi_rate={ex.get('isi_rate', 0.0):.4f}"
        )
        if ex.get("waveform_corr") is not None:
            lines.append(
                f"- waveform_corr={_safe_float(ex.get('waveform_corr')):.4f} merged_isi_rate={_safe_float(ex.get('merged_isi_rate')):.4f}"
            )
        lines.append(f"- GT action: {gt_action}")
        if reason:
            lines.append(f"- GT reasoning: {reason}")
    return "\n".join(lines)
