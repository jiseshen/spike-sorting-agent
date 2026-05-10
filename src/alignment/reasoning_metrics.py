"""
Reasoning alignment metrics.

Measures similarity between student rationale and teacher feedback using:
  1. Cosine similarity of sentence embeddings (requires sentence-transformers)
  2. LLM-as-judge scoring via src.agent.api (optional)
"""

from __future__ import annotations

from typing import List, Optional


def compute_reasoning_alignment(
    student_rationales: List[str],
    teacher_feedbacks: List[str],
    method: str = "cosine",
    llm_judge_model: Optional[str] = None,
    provider: str = "gpt4o",
    use_mock: bool = False,
) -> dict:
    """
    Compute reasoning alignment between student rationales and teacher feedback.

    Args:
        student_rationales: List of student rationale strings (one per step).
        teacher_feedbacks: List of teacher feedback strings (one per step).
        method: "cosine" (embedding similarity) or "llm_judge".
        llm_judge_model: Model to use if method == "llm_judge".
        provider: VLM provider for LLM judge calls.
        use_mock: Return placeholder scores without API calls.

    Returns:
        Dict with keys: method, mean_score, per_step_scores.
    """
    if len(student_rationales) != len(teacher_feedbacks):
        raise ValueError("student_rationales and teacher_feedbacks must have the same length.")

    if not student_rationales:
        return {"method": method, "mean_score": 0.0, "per_step_scores": []}

    if method == "cosine":
        scores = _cosine_similarity_scores(student_rationales, teacher_feedbacks, use_mock)
    elif method == "llm_judge":
        scores = _llm_judge_scores(
            student_rationales,
            teacher_feedbacks,
            model=llm_judge_model or "gpt-4o",
            provider=provider,
            use_mock=use_mock,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'cosine' or 'llm_judge'.")

    return {
        "method": method,
        "mean_score": sum(scores) / len(scores),
        "per_step_scores": scores,
    }


def _cosine_similarity_scores(
    texts_a: List[str],
    texts_b: List[str],
    use_mock: bool = False,
) -> List[float]:
    """Compute cosine similarity using sentence-transformers embeddings."""
    if use_mock:
        return [0.5] * len(texts_a)

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for cosine similarity. "
            "Install with: pip install sentence-transformers"
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_a = model.encode(texts_a, normalize_embeddings=True)
    emb_b = model.encode(texts_b, normalize_embeddings=True)

    scores = (emb_a * emb_b).sum(axis=1).tolist()
    return [float(s) for s in scores]


def _llm_judge_scores(
    student_rationales: List[str],
    teacher_feedbacks: List[str],
    model: str,
    provider: str,
    use_mock: bool = False,
) -> List[float]:
    """
    Use an LLM to judge the alignment between student reasoning and teacher feedback.

    Scores each pair on a 0-1 scale.
    """
    if use_mock:
        return [0.5] * len(student_rationales)

    from src.agent.api import call_vlm

    scores: list[float] = []
    for student_text, teacher_text in zip(student_rationales, teacher_feedbacks):
        prompt = (
            "Rate the alignment between a student's spike sorting reasoning and "
            "the expert teacher's feedback on a scale from 0.0 to 1.0.\n\n"
            f"Student reasoning:\n{student_text}\n\n"
            f"Teacher feedback:\n{teacher_text}\n\n"
            "Reply with a single float between 0.0 (no alignment) and 1.0 (perfect alignment). "
            "No explanation, just the number."
        )
        response = call_vlm(
            prompt=prompt,
            images=[],
            model=model,
            provider=provider,
            use_mock=use_mock,
            temperature=0.0,
        )
        try:
            score = float(response.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5
        scores.append(score)

    return scores
