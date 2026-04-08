"""
Scoring strategies for Life Test questions.

Cost order (cheapest first):
  1. exact        — string comparison, free
  2. semantic     — local sentence-transformer embeddings, free
  3. rubric       — one LLM (Haiku) call per criterion, ~$0.001 per question

The runner calls Scorer.score() which dispatches automatically.
"""
from __future__ import annotations
import os
from runner.models import Question, QuestionScore, RubricPoint

# Lazy imports so the runner works even when optional deps aren't installed
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _embedder = False  # mark as unavailable
    return _embedder if _embedder is not False else None


class Scorer:
    """Dispatches to the correct scoring strategy."""

    @staticmethod
    def score(question: Question, raw_answer: str) -> QuestionScore:
        strategy = question.scoring.type
        if strategy == "exact":
            return _exact_score(question, raw_answer)
        elif strategy == "exact_or_semantic":
            return _semantic_score(question, raw_answer)
        elif strategy == "rubric":
            return _rubric_score(question, raw_answer)
        raise ValueError(f"Unknown scoring type: {strategy}")


# ── Exact ─────────────────────────────────────────────────────────────────────

def _exact_score(question: Question, raw_answer: str) -> QuestionScore:
    passed = question.expected_answer.lower().strip() in raw_answer.lower().strip()
    return QuestionScore(
        question_id=question.question_id,
        text=question.text,
        difficulty=question.difficulty,
        possible=question.points,
        earned=question.points if passed else 0,
        raw_answer=raw_answer,
        passed=passed,
    )


# ── Semantic ──────────────────────────────────────────────────────────────────

def _semantic_score(question: Question, raw_answer: str) -> QuestionScore:
    threshold = question.scoring.threshold
    similarity = _cosine_similarity(question.expected_answer, raw_answer)
    passed = similarity >= threshold
    return QuestionScore(
        question_id=question.question_id,
        text=question.text,
        difficulty=question.difficulty,
        possible=question.points,
        earned=question.points if passed else 0,
        raw_answer=raw_answer,
        passed=passed,
        breakdown=[{"similarity": round(similarity, 4), "threshold": threshold}],
    )


def _cosine_similarity(a: str, b: str) -> float:
    embedder = _get_embedder()
    if embedder is None:
        # Fallback: substring overlap ratio when embeddings unavailable
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words:
            return 0.0
        return len(a_words & b_words) / len(a_words)
    import numpy as np
    vecs = embedder.encode([a, b], convert_to_numpy=True)
    cos = float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9))
    return cos


# ── Rubric ────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are a strict but fair grader evaluating an AI assistant's answer.

Question asked: {question}
Criterion to check: {criterion}
AI's answer: {answer}

Does the AI's answer satisfy the criterion? Answer with exactly one word: YES or NO"""


def _rubric_score(question: Question, raw_answer: str) -> QuestionScore:
    breakdown = []
    earned = 0

    for rubric_point in question.scoring.rubric_points:
        met = _judge_criterion(question.text, rubric_point.criterion, raw_answer)
        pts = rubric_point.points if met else 0
        earned += pts
        breakdown.append({
            "criterion": rubric_point.criterion,
            "possible": rubric_point.points,
            "earned": pts,
            "met": met,
        })

    return QuestionScore(
        question_id=question.question_id,
        text=question.text,
        difficulty=question.difficulty,
        possible=question.points,
        earned=earned,
        raw_answer=raw_answer,
        passed=earned >= question.points * 0.6,  # 60% threshold for "pass"
        breakdown=breakdown,
    )


def _judge_criterion(question_text: str, criterion: str, answer: str) -> bool:
    """Call a cheap LLM judge to evaluate a single criterion. Returns True/False."""
    prompt = _JUDGE_PROMPT.format(
        question=question_text,
        criterion=criterion,
        answer=answer,
    )
    response = _call_judge(prompt)
    return response.strip().upper().startswith("YES")


def _call_judge(prompt: str) -> str:
    """Call the judge model. Defaults to Claude Haiku for minimum cost."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
