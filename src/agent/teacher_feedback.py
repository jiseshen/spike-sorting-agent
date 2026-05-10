"""
Prompt builders for simulated teacher feedback.
"""

from typing import Dict, Optional


def build_teacher_feedback_prompt(
    *,
    channel: str,
    step: int,
    action_type: str,
    context_summary: Dict[str, object],
    student_action: str,
    student_rationale: str,
    human_action: str,
    human_reasoning: str,
    merge_target_id: Optional[int] = None,
) -> str:
    """
    Build a concise teacher prompt that critiques student action/reasoning.

    Important:
    - Includes the same observation context as the student (metrics + images passed separately)
    - Does NOT include student policy instructions
    - Includes human reference action/reasoning for supervision, but asks teacher not to reveal it
    """
    merge_target_line = (
        f"- Merge target cluster: {merge_target_id}\n" if merge_target_id is not None else ""
    )

    return f"""You are simulating expert human feedback for spike sorting.
Give brief feedback (1-2 sentences). Be direct and specific.

Task context:
- Channel: {channel}
- Step: {step}
- Action type: {action_type}
- Cluster under review: {context_summary.get("cluster_id")}
{merge_target_line}Student-observed numeric context (images are attached separately):
{_format_context_summary(context_summary)}

Student output:
- Action: {student_action}
- Reasoning: {student_rationale}

Reference annotation for teacher-only calibration (DO NOT quote or reveal directly):
- Reference action: {human_action}
- Reference reasoning: {human_reasoning}

Output rules:
1) If student is correct: short approval.
2) If student is wrong/partly wrong: only point out the incorrect part.
3) Do not reveal the reference action/reasoning verbatim.
4) Keep it brief and practical.
"""


def _format_context_summary(context: Dict[str, object]) -> str:
    lines = []
    for k, v in context.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)
