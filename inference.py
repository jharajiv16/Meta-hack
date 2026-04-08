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
# ------------------------------------------------------------------ #

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "total_reward": sum(rewards),
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)


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
        print(f"\n{'='*60}", flush=True)
        print(f"  Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        score = await run_task(task_name, use_llm=use_llm)
        scores[task_name] = score
        print(f"\n  >> Task '{task_name}' score: {score:.4f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("  FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for name, s in scores.items():
        status = "✅ PASS" if s >= SUCCESS_SCORE_THRESHOLD else "❌ FAIL"
        print(f"  {name:20s} : {s:.4f}  {status}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':20s} : {avg:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())