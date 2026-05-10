"""
GroundTruthTeacher: produces calibrated feedback at each trajectory step.

Wraps the existing src.agent.teacher_feedback prompt builder and enriches it
with ground-truth context from the oracle (gt_action + gt_reasoning), plus
the per-setting teacher_criteria that define what this "lab style" values.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.agent.api import call_vlm
from src.agent.teacher_feedback import build_teacher_feedback_prompt
from src.simulate.setting import SettingConfig


class GroundTruthTeacher:
    """
    Produces teacher feedback for a single trajectory step.

    Args:
        cfg: SettingConfig containing teacher_criteria and teacher_style.
        teacher_model: Model name to use for teacher calls.
        provider: VLM provider string (e.g. "gpt4o", "claude").
        use_mock: If True, return placeholder feedback without API calls.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        cfg: SettingConfig,
        teacher_model: str,
        provider: str = "gpt4o",
        use_mock: bool = False,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.teacher_model = teacher_model
        self.provider = provider
        self.use_mock = use_mock
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def get_feedback(
        self,
        *,
        channel_id: str,
        step: int,
        action_type: str,
        context_summary: Dict[str, object],
        student_action: str,
        student_rationale: str,
        gt_action: str,
        gt_reasoning: str,
        images: List,
        merge_target_id: Optional[int] = None,
        output_dir=None,
    ) -> str:
        """
        Call the teacher model and return its feedback string.

        The teacher sees:
          - Same cluster context/images the student saw
          - Student's action + rationale
          - GT reference action + reasoning (for calibration; not to be quoted verbatim)
          - Setting-specific teacher criteria (injected into prompt preamble)
        """
        criteria_preamble = self._format_criteria_preamble()

        prompt = build_teacher_feedback_prompt(
            channel=channel_id,
            step=step,
            action_type=action_type,
            context_summary=context_summary,
            student_action=student_action,
            student_rationale=student_rationale,
            human_action=gt_action,
            human_reasoning=gt_reasoning,
            merge_target_id=merge_target_id,
        )
        prompt = criteria_preamble + prompt

        if self.use_mock:
            return f"[MOCK TEACHER] GT={gt_action}. Student said {student_action}."

        feedback = call_vlm(
            prompt=prompt,
            images=images,
            model=self.teacher_model,
            provider=self.provider,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

        if output_dir is not None:
            from pathlib import Path
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"teacher_prompt_step{step:04d}.txt").write_text(prompt)
            (out / f"teacher_feedback_step{step:04d}.txt").write_text(feedback)

        return feedback

    def _format_criteria_preamble(self) -> str:
        c = self.cfg
        style_desc = {
            "strict": "strict quality standards (low ISI tolerance, high SNR requirement)",
            "liberal": "liberal quality standards (tolerant of mild ISI violations)",
            "snr_only": "SNR-focused standards (prioritize signal amplitude over ISI purity)",
        }.get(c.teacher_style, c.teacher_style)

        return (
            f"[Teacher criteria — {c.teacher_style} lab style ({style_desc})]\n"
            f"  min_spike_count: {c.min_spike_count}\n"
            f"  max_isi_violation_rate: {c.max_isi_violation_rate:.4f}\n"
            f"  require_biphasic: {c.require_biphasic}\n"
            f"  min_snr: {c.min_snr}\n\n"
        )
