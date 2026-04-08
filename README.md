---
title: AI Startup Founder Simulator
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
---

# 🚀 AI Startup Founder Simulator

An **OpenEnv-compliant** reinforcement learning environment that simulates the life of an AI startup founder. Agents must make strategic decisions about hiring, product development, marketing, and fundraising to survive and grow over 60 simulated months.

## 🎯 Real-World Utility

Training AI agents on business strategy is a genuine research challenge. This environment models the complex trade-offs every startup founder faces:

- **Cash management** vs. growth investment
- **Team building** vs. burn rate control
- **Product quality** vs. speed to market
- **Technical debt** vs. feature velocity

Unlike toy environments, every decision has cascading consequences across multiple metrics, requiring long-horizon planning.

---

## 📋 Action Space

The agent chooses **one action per month** from 8 options:

| Action | Cost | Effect |
|---|---|---|
| `hire_engineer` | ₹5,000 | +1 team member, +quality, +tech debt |
| `fire_employee` | Free | -1 team member, **-20% morale** |
| `build_feature` | ₹1,500 | +quality (scaled by morale & debt) |
| `run_marketing` | ₹2,000 | +reach (scaled by market sentiment) |
| `raise_funding` | Free | 20% chance of ₹20k+ injection |
| `do_nothing` | Free | -2% tech debt (natural recovery) |
| `pivot` | ₹15,000 | Reset quality to 0.2, **clear all tech debt** |
| `train_team` | ₹3,000 | +15% morale |

## 📊 Observation Space

Each step returns a `StartupObservation` with these fields:

| Field | Type | Range | Description |
|---|---|---|---|
| `month` | int | 0–60 | Current simulation month |
| `cash` | float | 0–∞ | Available capital (₹) |
| `team_size` | int | 1–∞ | Number of employees |
| `product_quality` | float | 0–1 | Product quality score |
| `marketing_reach` | float | 0–1 | Market penetration |
| `revenue` | float | 0–∞ | Monthly recurring revenue (₹) |
| `burn_rate` | float | 0–∞ | Monthly expenses (₹) |
| `tech_debt` | float | 0–1 | Accumulated technical debt |
| `team_morale` | float | 0–1 | Team happiness/productivity |
| `market_sentiment` | float | 0.5–1.5 | External market conditions |
| `events` | list[str] | — | Random events that occurred |
| `done` | bool | — | Whether the episode ended |
| `reward` | float | — | Reward signal for this step |

## 🏆 Tasks (Easy → Medium → Hard)

### Task 1: Runway (Easy)
**Objective:** Survive for at least 12 months without going bankrupt.
**Grading:** `score = months_survived / 12` (clamped to 0–1).

### Task 2: Market Fit (Medium)
**Objective:** Achieve ₹100,000 in monthly revenue at any point.
**Grading:** `score = best_revenue / 100,000` (clamped to 0–1).

### Task 3: Unicorn (Hard)
**Objective:** Achieve a valuation of ₹1,000,000 while keeping tech debt low.
**Valuation** = `revenue × 10 + product_quality × 50,000`, penalised by tech debt.
**Grading:** `score = best_effective_valuation / 1,000,000` (clamped to 0–1).

---

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/jharajiv16/Meta-hack.git
cd Meta-hack

# Install dependencies
pip install -e .

# Run the server
python -m server.app
# or equivalently:
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
# Rule-based baseline (no API key needed)
python inference.py

# With LLM (set environment variables)
export API_BASE_URL="https://your-api-endpoint/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python inference.py
```

### Docker

```bash
docker build -t ai-startup-simulator .
docker run -p 7860:7860 ai-startup-simulator
```

---

## 📈 Baseline Scores (Rule-Based Agent)

| Task | Score | Status |
|---|---|---|
| runway (Easy) | ~1.00 | ✅ Pass |
| market_fit (Medium) | ~0.45 | ✅ Pass |
| unicorn (Hard) | ~0.12 | ❌ Challenging |

*Scores measured with `seed=42` for reproducibility.*

---

## 🏗️ Project Structure

```
├── server/
│   ├── __init__.py
│   ├── app.py          # OpenEnv server (FastAPI + WebSocket)
│   ├── env.py          # Environment (reset/step/state)
│   ├── models.py       # Pydantic typed models (Action/Observation/State)
│   ├── tasks.py        # 3 graded tasks with deterministic graders
│   └── openenv.yaml    # Environment configuration
├── inference.py        # Baseline inference script (OpenAI client)
├── openenv.yaml        # Root config (copied into server/)
├── pyproject.toml      # Project metadata & dependencies
├── Dockerfile          # Container build
├── requirements.txt    # Pip requirements
└── README.md           # This file
```

## 🎲 Reward Function

The reward function provides continuous signal (not sparse):

```
reward = (revenue/1000) × 15
       + product_quality × 100
       + marketing_reach × 50
       + team_morale × 40
       - tech_debt × 60
       - (burn_rate/1000) × 5
       + 20 (survival bonus)
```

Bankruptcy incurs a **-5000 penalty**.

---

## 📜 License

MIT License
