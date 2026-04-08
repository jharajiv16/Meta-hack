"""
Pydantic typed models for the AI Startup Founder Simulator.
Implements the OpenEnv spec: Action, Observation, State.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core import Action, Observation, State


class StartupAction(Action):
    """A single strategic decision the agent makes each month."""
    action: str = Field(
        description=(
            "The action to take. One of: "
            "'hire_engineer', 'fire_employee', 'build_feature', "
            "'run_marketing', 'raise_funding', 'do_nothing', 'pivot', 'train_team'"
        )
    )


class StartupObservation(Observation):
    """The observable state returned to the agent after each step."""
    month: int = Field(default=0, description="Current month (0-indexed)")
    cash: float = Field(default=50000.0, description="Available cash (₹)")
    team_size: int = Field(default=1, description="Number of employees")
    product_quality: float = Field(default=0.1, description="Product quality score 0-1")
    marketing_reach: float = Field(default=0.1, description="Marketing reach score 0-1")
    revenue: float = Field(default=0.0, description="Monthly recurring revenue (₹)")
    burn_rate: float = Field(default=1200.0, description="Monthly burn rate (₹)")
    tech_debt: float = Field(default=0.1, description="Technical debt level 0-1")
    team_morale: float = Field(default=1.0, description="Team morale 0-1")
    market_sentiment: float = Field(default=1.0, description="Market sentiment 0.5-1.5")
    events: List[str] = Field(default_factory=list, description="Events that occurred this step")


class StartupState(State):
    """Full internal state of the simulator (includes private details)."""
    month: int = 0
    cash: float = 50000.0
    team_size: int = 1
    product_quality: float = 0.1
    marketing_reach: float = 0.1
    revenue: float = 0.0
    burn_rate: float = 1200.0
    tech_debt: float = 0.1
    team_morale: float = 1.0
    market_sentiment: float = 1.0
    events: List[str] = Field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
