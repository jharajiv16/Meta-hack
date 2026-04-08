"""
AI Startup Founder Simulator — OpenEnv-compliant Environment.

Implements the full OpenEnv interface:
  - reset(seed) → StartupObservation
  - step(action) → StartupObservation
  - state (property) → StartupState
"""
from __future__ import annotations

import os
import random
from typing import Any, Optional

import yaml
from openenv.core import Environment

from .models import StartupAction, StartupObservation, StartupState

ACTION_MAP = {
    "hire_engineer": 0,
    "fire_employee": 1,
    "build_feature": 2,
    "run_marketing": 3,
    "raise_funding": 4,
    "do_nothing": 5,
    "pivot": 6,
    "train_team": 7,
}


class StartupEnv(Environment[StartupAction, StartupObservation, StartupState]):
    """
    Simulates the life of an AI startup founder.
    The agent must manage cash, team, product, and marketing
    to survive and grow over 60 months.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)

        if config_path is None:
            # Look for openenv.yaml relative to this file, then cwd
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(here, "openenv.yaml"),
                os.path.join(here, "..", "openenv.yaml"),
                "openenv.yaml",
            ]
            for c in candidates:
                if os.path.exists(c):
                    config_path = c
                    break
            else:
                raise FileNotFoundError("openenv.yaml not found")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["environment"]

        # Internal mutable state
        self._month = 0
        self._cash = float(self.config["initial_cash"])
        self._team_size = 1
        self._product_quality = 0.1
        self._marketing_reach = 0.1
        self._revenue = 0.0
        self._burn_rate = float(self.config["burn_rate_per_employee"])
        self._tech_debt = 0.1
        self._team_morale = 1.0
        self._market_sentiment = 1.0
        self._events: list[str] = []
        self._terminated = False
        self._truncated = False
        self._step_count = 0

    # ------------------------------------------------------------------ #
    # OpenEnv interface
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StartupObservation:
        if seed is not None:
            random.seed(seed)

        self._month = 0
        self._cash = float(self.config["initial_cash"])
        self._team_size = 1
        self._product_quality = 0.1
        self._marketing_reach = 0.1
        self._revenue = 0.0
        self._burn_rate = float(self.config["burn_rate_per_employee"])
        self._tech_debt = 0.1
        self._team_morale = 1.0
        self._market_sentiment = 1.0
        self._events = []
        self._terminated = False
        self._truncated = False
        self._step_count = 0

        return self._make_observation(reward=0.0)

    def step(
        self,
        action: StartupAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StartupObservation:
        self._month += 1
        self._step_count += 1
        self._events = []

        action_idx = ACTION_MAP.get(action.action, 5)
        self._apply_action(action_idx)
        self._apply_monthly_updates()
        self._apply_random_events()
        self._cash += self._revenue

        reward = self._calculate_reward()

        # Termination
        if self._cash <= 0:
            self._terminated = True
            reward -= 5000
            self._events.append("BANKRUPTCY. Game over.")

        if self._month >= self.config["max_months"]:
            self._truncated = True

        done = self._terminated or self._truncated
        return self._make_observation(reward=reward, done=done)

    @property
    def state(self) -> StartupState:
        return StartupState(
            episode_id=None,
            step_count=self._step_count,
            month=self._month,
            cash=self._cash,
            team_size=self._team_size,
            product_quality=self._product_quality,
            marketing_reach=self._marketing_reach,
            revenue=self._revenue,
            burn_rate=self._burn_rate,
            tech_debt=self._tech_debt,
            team_morale=self._team_morale,
            market_sentiment=self._market_sentiment,
            events=list(self._events),
            terminated=self._terminated,
            truncated=self._truncated,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _make_observation(self, reward: float, done: bool = False) -> StartupObservation:
        return StartupObservation(
            month=self._month,
            cash=self._cash,
            team_size=self._team_size,
            product_quality=self._product_quality,
            marketing_reach=self._marketing_reach,
            revenue=self._revenue,
            burn_rate=self._burn_rate,
            tech_debt=self._tech_debt,
            team_morale=self._team_morale,
            market_sentiment=self._market_sentiment,
            events=list(self._events),
            done=done,
            reward=reward,
        )

    def _apply_action(self, action: int) -> None:
        cfg = self.config
        if action == 0:  # hire_engineer
            if self._cash >= cfg["hiring_cost"]:
                self._cash -= cfg["hiring_cost"]
                self._team_size += 1
                gain = 0.05 * (1.0 - self._tech_debt) * self._team_morale
                self._product_quality = min(1.0, self._product_quality + gain)
                self._tech_debt = min(1.0, self._tech_debt + cfg["tech_debt_increment_per_hire"])
                self._events.append("Hired an engineer. Tech debt increased.")
        elif action == 1:  # fire_employee
            if self._team_size > 1:
                self._team_size -= 1
                self._team_morale = max(0.0, self._team_morale - 0.2)
                self._events.append("Fired an employee. Morale dropped.")
        elif action == 2:  # build_feature
            if self._cash >= cfg["feature_cost"]:
                self._cash -= cfg["feature_cost"]
                gain = 0.1 * (1.0 - self._tech_debt) * self._team_morale
                self._product_quality = min(1.0, self._product_quality + gain)
                self._tech_debt = min(1.0, self._tech_debt + cfg["tech_debt_increment_per_feature"])
                self._events.append("Built a feature. Tech debt increased.")
        elif action == 3:  # run_marketing
            if self._cash >= cfg["marketing_cost"]:
                self._cash -= cfg["marketing_cost"]
                reach_gain = 0.15 * self._market_sentiment
                self._marketing_reach = min(1.0, self._marketing_reach + reach_gain)
                self._events.append("Marketing campaign successful.")
        elif action == 4:  # raise_funding
            prob = cfg["funding_probability"] * self._market_sentiment
            if random.random() < prob:
                amount = max(20000, (self._revenue * 15) + (self._product_quality * 100000))
                self._cash += amount
                self._events.append(f"Raised ₹{amount:,.0f} in funding!")
            else:
                self._events.append("Funding pitch failed.")
        elif action == 5:  # do_nothing
            self._tech_debt = max(0.0, self._tech_debt - 0.02)
            self._events.append("Idle month. Tech debt slightly reduced.")
        elif action == 6:  # pivot
            if self._cash >= cfg["pivot_cost"]:
                self._cash -= cfg["pivot_cost"]
                self._product_quality = 0.2
                self._tech_debt = 0.0
                self._events.append("Pivoted! Quality reset, Tech Debt cleared.")
        elif action == 7:  # train_team
            if self._cash >= cfg["training_cost"]:
                self._cash -= cfg["training_cost"]
                self._team_morale = min(1.0, self._team_morale + cfg["morale_recovery_per_training"])
                self._events.append("Trained team. Morale boosted.")

    def _apply_monthly_updates(self) -> None:
        cfg = self.config
        # Morale decay
        self._team_morale = max(0.0, self._team_morale - cfg["morale_decay_per_month"])
        if self._cash < self._burn_rate * 2:
            self._team_morale = max(0.0, self._team_morale - 0.05)
            self._events.append("Low runway is hurting team morale!")
        # Burn rate
        self._burn_rate = self._team_size * cfg["burn_rate_per_employee"]
        self._cash -= self._burn_rate
        # Revenue
        self._revenue = (self._product_quality * self._marketing_reach * 10000) + (self._team_size * 100)

    def _apply_random_events(self) -> None:
        if random.random() < self.config["random_event_probability"]:
            roll = random.random()
            if roll < 0.3:
                self._market_sentiment = max(0.5, self._market_sentiment - 0.3)
                self._revenue *= 0.5
                self._events.append("MARKET CRASH! Sentiment plummeted.")
            elif roll < 0.6:
                self._marketing_reach = min(1.0, self._marketing_reach + 0.3)
                self._market_sentiment = min(1.5, self._market_sentiment + 0.2)
                self._events.append("VIRAL GROWTH! Marketing exploded.")
            elif roll < 0.8:
                self._marketing_reach = max(0.0, self._marketing_reach - 0.2)
                self._events.append("COMPETITOR LAUNCH! Market share lost.")
            else:
                self._market_sentiment = min(1.5, self._market_sentiment + 0.3)
                self._events.append("ECONOMIC BOOM! Investors are bullish.")

    def _calculate_reward(self) -> float:
        r = 0.0
        r += (self._revenue / 1000) * 15
        r += self._product_quality * 100
        r += self._marketing_reach * 50
        r += self._team_morale * 40
        r -= self._tech_debt * 60
        r -= (self._burn_rate / 1000) * 5
        r += 20  # survival bonus
        return r
