"""Shared data models for the Life Test runner."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricPoint:
    criterion: str
    points: int


@dataclass
class ScoringConfig:
    type: str  # "exact" | "exact_or_semantic" | "rubric"
    threshold: float = 0.85
    rubric_points: list[RubricPoint] = field(default_factory=list)


@dataclass
class Question:
    question_id: str
    text: str
    expected_answer: str
    difficulty: int       # 1–5
    points: int
    scoring: ScoringConfig


@dataclass
class Event:
    event_id: str
    event_type: str
    timestamp: str
    payload: dict[str, Any]


@dataclass
class QuestionScore:
    question_id: str
    text: str
    difficulty: int
    possible: int
    earned: int
    raw_answer: str
    passed: bool
    breakdown: list[dict] | None = None


@dataclass
class PhaseResult:
    phase_id: str
    phase_type: str  # "event_section" | "question_section"
    scores: list[QuestionScore] = field(default_factory=list)
    ingestion_entries: list[dict] = field(default_factory=list)


@dataclass
class SituationResult:
    situation_id: str
    system_id: str
    total_possible: int = 0
    total_earned: int = 0
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def percent(self) -> float:
        if self.total_possible == 0:
            return 0.0
        return round(100 * self.total_earned / self.total_possible, 1)

    def by_difficulty(self) -> dict[int, dict]:
        """Aggregate scores grouped by difficulty level."""
        groups: dict[int, dict] = {}
        for phase in self.phases:
            for score in phase.scores:
                d = score.difficulty
                if d not in groups:
                    groups[d] = {"possible": 0, "earned": 0}
                groups[d]["possible"] += score.possible
                groups[d]["earned"] += score.earned
        for d, g in groups.items():
            g["percent"] = round(100 * g["earned"] / g["possible"], 1) if g["possible"] else 0.0
        return groups
