#!/usr/bin/env python3
"""
Baseline inference script for the AI Startup Founder Simulator.

Uses the OpenAI API client to run a model against the environment.
Emits structured stdout logs: [START], [STEP], [END].

Required environment variables:
  API_BASE_URL  — The API endpoint for the LLM.
  MODEL_NAME    — The model identifier to use for inference.
  HF_TOKEN      — Your Hugging Face / API key (used as OPENAI_API_KEY).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ------------------------------------------------------------------ #
# Configuration from environment
# ------------------------------------------------------------------ #
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "sk-placeholder"

IMAGE_NAME = os.environ.get("DOCKER_IMAGE", "ai-startup-simulator:latest")
BENCHMARK = "ai-startup-simulator"
MAX_STEPS = 60          # Max months in simulation
MAX_TOTAL_REWARD = MAX_STEPS * 300  # Rough upper bound on cumulative reward
SUCCESS_SCORE_THRESHOLD = 0.3

VALID_ACTIONS = [
    "hire_engineer", "fire_employee", "build_feature",
    "run_marketing", "raise_funding", "do_nothing",
    "pivot", "train_team",
]

TASK_NAMES = ["runway", "market_fit", "unicorn"]


# ------------------------------------------------------------------ #
# Structured logging helpers (MANDATORY FORMAT)
# The hackathon validator searches for literal [START], [STEP], [END]
# markers in stdout. Each marker must appear on its own line.
# ------------------------------------------------------------------ #

def log_start(task: str, env: str, model: str) -> None:
    data = json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    })
    print(f"[START] {data}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    data = json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    })
    print(f"[STEP] {data}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    data = json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "total_reward": sum(rewards),
        "rewards": rewards,
        "timestamp": time.time(),
    })
    print(f"[END] {data}", flush=True)



# ------------------------------------------------------------------ #
# LLM interaction
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an expert AI startup founder. You are playing a business simulation game.
Each turn you must choose EXACTLY ONE action from this list:
  hire_engineer, fire_employee, build_feature, run_marketing,
  raise_funding, do_nothing, pivot, train_team

Respond with ONLY the action name, nothing else. No explanation, no punctuation.

Strategy tips:
- Keep cash above 20000 to survive
- Build features to improve product quality
- Run marketing to increase reach and revenue
- Hire engineers early to build faster
- Train team to keep morale high
- Raise funding when cash is low
- Pivot only when tech debt is very high and quality is low
"""


def get_model_action(
    client: OpenAI,
    step: int,
    observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> str:
    """Ask the LLM to choose the next action based on observation."""
    obs_summary = (
        f"Month {observation.get('month', 0)}: "
        f"Cash=₹{observation.get('cash', 0):,.0f}, "
        f"Team={observation.get('team_size', 1)}, "
        f"Quality={observation.get('product_quality', 0):.2f}, "
        f"Reach={observation.get('marketing_reach', 0):.2f}, "
        f"Revenue=₹{observation.get('revenue', 0):,.0f}, "
        f"Morale={observation.get('team_morale', 1):.2f}, "
        f"TechDebt={observation.get('tech_debt', 0):.2f}, "
        f"Sentiment={observation.get('market_sentiment', 1):.2f}"
    )
    events = observation.get("events", [])
    if events:
        obs_summary += f"\nEvents: {', '.join(events)}"

    user_msg = f"Step {step}. Last reward: {last_reward:+.2f}\n{obs_summary}\nChoose your action:"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower().replace(" ", "_")
        # Validate
        if raw in VALID_ACTIONS:
            return raw
        # Try fuzzy match
        for a in VALID_ACTIONS:
            if a in raw:
                return a
        print(f"[DEBUG] Invalid action from model: '{raw}', defaulting to 'do_nothing'", flush=True)
        return "do_nothing"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "do_nothing"


# ------------------------------------------------------------------ #
# Rule-based fallback agent (no LLM needed)
# ------------------------------------------------------------------ #

def rule_based_action(obs: Dict[str, Any]) -> str:
    """Simple heuristic agent as a reproducible baseline."""
    cash = obs.get("cash", 50000)
    quality = obs.get("product_quality", 0.1)
    reach = obs.get("marketing_reach", 0.1)
    morale = obs.get("team_morale", 1.0)
    team = obs.get("team_size", 1)
    tech_debt = obs.get("tech_debt", 0.1)

    # Survival first
    if cash < 15000:
        return "raise_funding"
    # Build team early
    if team < 3 and cash > 25000:
        return "hire_engineer"
    # Tech debt too high? pivot
    if tech_debt > 0.7 and cash > 20000:
        return "pivot"
    # Build product quality
    if quality < 0.5:
        return "build_feature"
    # Grow reach
    if reach < 0.4:
        return "run_marketing"
    # Keep morale up
    if morale < 0.5:
        return "train_team"
    # Default: build more
    if quality < 0.8:
        return "build_feature"
    return "run_marketing"


# ------------------------------------------------------------------ #
# Agent classes (used by Gradio UI in root app.py)
# ------------------------------------------------------------------ #

class RuleBasedAgent:
    """Wrapper class for the rule-based heuristic agent."""

    def get_action(self, obs: Dict[str, Any]) -> int:
        """Return action index (0-7) based on observation dict."""
        # Handle numpy array values from gym env
        def _val(x):
            try:
                return float(x[0])
            except (TypeError, IndexError):
                return float(x)

        clean_obs = {
            "cash": _val(obs.get("cash", 50000)),
            "product_quality": _val(obs.get("product_quality", 0.1)),
            "marketing_reach": _val(obs.get("marketing_reach", 0.1)),
            "team_morale": _val(obs.get("team_morale", 1.0)),
            "team_size": int(_val(obs.get("team_size", 1))),
            "tech_debt": _val(obs.get("tech_debt", 0.1)),
        }
        action_str = rule_based_action(clean_obs)
        action_map = {
            "hire_engineer": 0, "fire_employee": 1, "build_feature": 2,
            "run_marketing": 3, "raise_funding": 4, "do_nothing": 5,
            "pivot": 6, "train_team": 7,
        }
        return action_map.get(action_str, 5)


class PPOAgentWrapper:
    """Placeholder PPO agent — falls back to rule-based until a trained model is available."""

    def __init__(self, model_path: Optional[str] = None):
        self._fallback = RuleBasedAgent()
        self._model = None
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(model_path)
            except Exception:
                pass

    def get_action(self, obs: Dict[str, Any]) -> int:
        """Return action index (0-7)."""
        if self._model is not None:
            try:
                import numpy as np
                flat = np.array([
                    float(obs.get("cash", [50000])[0] if hasattr(obs.get("cash", 50000), '__getitem__') else obs.get("cash", 50000)),
                    float(obs.get("team_size", [1])[0] if hasattr(obs.get("team_size", 1), '__getitem__') else obs.get("team_size", 1)),
                    float(obs.get("product_quality", [0.1])[0] if hasattr(obs.get("product_quality", 0.1), '__getitem__') else obs.get("product_quality", 0.1)),
                    float(obs.get("marketing_reach", [0.1])[0] if hasattr(obs.get("marketing_reach", 0.1), '__getitem__') else obs.get("marketing_reach", 0.1)),
                    float(obs.get("revenue", [0])[0] if hasattr(obs.get("revenue", 0), '__getitem__') else obs.get("revenue", 0)),
                    float(obs.get("burn_rate", [1200])[0] if hasattr(obs.get("burn_rate", 1200), '__getitem__') else obs.get("burn_rate", 1200)),
                    float(obs.get("month", [0])[0] if hasattr(obs.get("month", 0), '__getitem__') else obs.get("month", 0)),
                    float(obs.get("tech_debt", [0.1])[0] if hasattr(obs.get("tech_debt", 0.1), '__getitem__') else obs.get("tech_debt", 0.1)),
                    float(obs.get("team_morale", [1.0])[0] if hasattr(obs.get("team_morale", 1.0), '__getitem__') else obs.get("team_morale", 1.0)),
                    float(obs.get("market_sentiment", [1.0])[0] if hasattr(obs.get("market_sentiment", 1.0), '__getitem__') else obs.get("market_sentiment", 1.0)),
                ], dtype=np.float32)
                action, _ = self._model.predict(flat, deterministic=True)
                return int(action)
            except Exception:
                pass
        return self._fallback.get_action(obs)


# ------------------------------------------------------------------ #
# Main async runner
# ------------------------------------------------------------------ #

async def run_task(task_name: str, use_llm: bool = False) -> float:
    """Run a single task and return the score."""
    from server.env import StartupEnv
    from server.models import StartupAction

    env = StartupEnv()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if use_llm else None

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME if use_llm else "rule-based")

    try:
        result = env.reset(seed=42)
        last_reward = 0.0

        # Collect trajectory for grading
        trajectory = [result]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = result.model_dump()

            if use_llm and client:
                action_str = get_model_action(client, step, obs_dict, last_reward, history)
            else:
                action_str = rule_based_action(obs_dict)

            action = StartupAction(action=action_str)
            result = env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str!r} -> reward {reward:+.2f}")
            trajectory.append(result)

            if done:
                break

        # Grade using task graders
        from server.tasks import get_tasks
        task_map = {t.name: t for t in get_tasks()}

        if task_name in task_map:
            score = task_map[task_name].grader(trajectory)
        else:
            # Fallback: use cumulative reward
            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Run all tasks and report scores."""
    use_llm = API_KEY not in ("sk-placeholder", "", None)

    if not use_llm:
        print("[INFO] No API key found. Using rule-based baseline agent.", flush=True)

    scores = {}
    for task_name in TASK_NAMES:
        try:
            score = await run_task(task_name, use_llm=use_llm)
            scores[task_name] = score
            print(f"  >> Task '{task_name}' score: {score:.4f}", flush=True)
        except Exception as exc:
            print(f"[ERROR] Task '{task_name}' failed: {exc}", flush=True)
            # Still emit START/END so the validator sees something
            log_start(task=task_name, env=BENCHMARK, model="error")
            log_end(success=False, steps=0, score=0.0, rewards=[])
            scores[task_name] = 0.0

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  AVERAGE : {avg:.4f}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        # Last-resort fallback: emit at least one START/END pair
        print(f"[START] {{\"type\":\"START\",\"task\":\"error\",\"env\":\"ai-startup-simulator\",\"model\":\"error\",\"timestamp\":{time.time()}}}", flush=True)
        print(f"[ERROR] Fatal error: {exc}", flush=True)
        print(f"[END] {{\"type\":\"END\",\"success\":false,\"steps\":0,\"score\":0.0,\"total_reward\":0.0,\"rewards\":[],\"timestamp\":{time.time()}}}", flush=True)
        sys.exit(1)