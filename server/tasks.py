"""
Three graded tasks for the AI Startup Founder Simulator.

Each task defines:
  - A concrete objective
  - A deterministic grader that scores 0.0–1.0
  - Clear difficulty progression: Easy → Medium → Hard
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from .models import StartupObservation


@dataclass
class Task:
    name: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    grader: Callable[[List[StartupObservation]], float]


# ------------------------------------------------------------------ #
# Grader functions
# ------------------------------------------------------------------ #

def grade_runway(trajectory: List[StartupObservation]) -> float:
    """
    EASY — Initial Runway.
    Objective: Survive for at least 12 months without going bankrupt.
    Score: months_survived / 12, clamped to [0, 1].
    """
    months_survived = len(trajectory)
    return min(1.0, months_survived / 12.0)


def grade_market_fit(trajectory: List[StartupObservation]) -> float:
    """
    MEDIUM — Market Fit.
    Objective: Achieve ₹100,000 in monthly revenue at any point during the episode.
    Score: best_revenue / 100_000, clamped to [0, 1].
    """
    if not trajectory:
        return 0.0
    best_revenue = max(obs.revenue for obs in trajectory)
    return min(1.0, best_revenue / 100_000.0)


def grade_unicorn(trajectory: List[StartupObservation]) -> float:
    """
    HARD — Unicorn Growth.
    Objective: Achieve a valuation of ₹1,000,000.
    Valuation = revenue * 10 + product_quality * 50,000
    Also penalises high tech debt (sustainable growth required).
    Score: best_effective_valuation / 1_000_000, clamped to [0, 1].
    """
    if not trajectory:
        return 0.0

    def valuation(obs: StartupObservation) -> float:
        raw = (obs.revenue * 10) + (obs.product_quality * 50_000)
        # Penalise tech debt — incentivises sustainable growth
        return raw * (1.0 - obs.tech_debt * 0.5)

    best = max(valuation(obs) for obs in trajectory)
    return min(1.0, best / 1_000_000.0)


# ------------------------------------------------------------------ #
# Task registry
# ------------------------------------------------------------------ #

def get_tasks() -> List[Task]:
    """Return all tasks sorted by difficulty."""
    return [
        Task(
            name="runway",
            difficulty="easy",
            description=(
                "Survive for at least 12 months without going bankrupt. "
                "Score = months_survived / 12."
            ),
            grader=grade_runway,
        ),
        Task(
            name="market_fit",
            difficulty="medium",
            description=(
                "Achieve ₹100,000 in monthly revenue at any point during the episode. "
                "Score = best_revenue / 100,000."
            ),
            grader=grade_market_fit,
        ),
        Task(
            name="unicorn",
            difficulty="hard",
            description=(
                "Achieve a valuation of ₹1,000,000 (revenue×10 + quality×50k) "
                "while keeping tech debt low. "
                "Score = best_effective_valuation / 1,000,000."
            ),
            grader=grade_unicorn,
        ),
    ]
