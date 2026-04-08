---
title: Meta Hack Simulator
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.11.0
python_version: "3.10"
app_file: app.py
---

# 🚀 AI Startup Founder Simulator

A reinforcement learning environment and interactive dashboard for simulating the journey of an AI startup founder.

## 🎯 Project Overview

This project provides a custom Gymnasium environment where an AI agent (or a human player) takes on the role of a startup founder. You'll make critical business decisions each month, including:

- **Hiring & Firing:** Manage your team size and burn rate.
- **Product Development:** Build features to improve product quality.
- **Marketing:** Launch campaigns to increase your reach.
- **Fundraising:** Pitch to investors for cash injections.
- **Survival:** Navigate random market crashes and viral growth events.

## 🛠 Project Structure

- `env.py`: The `StartupEnv` Gymnasium class.
- `tasks.py`: Evaluation logic for Easy (Survival), Medium (Revenue), and Hard (Valuation) tasks.
- `inference.py`: Rule-based and PPO agent implementations.
- `app.py`: Gradio-based interactive dashboard.
- `test_env.py`: Basic validation script for the environment.
- `openenv.yaml`: Configuration for environment parameters.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Containerization setup.

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Simulator (UI)

```bash
python app.py
```

This will launch a Gradio interface where you can manually step through the simulation or watch an AI agent run.

### 3. Run Validation Tests

```bash
python test_env.py
```

## 🧠 Environment Design

- **State:** Cash, Team Size, Product Quality, Marketing Reach, Revenue, Burn Rate, and Month.
- **Actions:** Discrete actions for founder decisions.
- **Reward:** Weighted based on revenue growth, product quality, and survival stability.

## 📦 Deployment

This project is compatible with **Hugging Face Spaces**. Simply upload the files and select the Gradio SDK.

Alternatively, use the provided `Dockerfile`:

```bash
docker build -t startup-simulator .
docker run -p 7860:7860 startup-simulator
```
