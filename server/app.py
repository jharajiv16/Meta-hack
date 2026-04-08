"""
AI Startup Founder Simulator — Server entry point.

A clean FastAPI app with:
  - GET /          → HTML dashboard
  - GET /health    → Health check
  - POST /reset    → Reset environment
  - POST /step     → Take a step
  - GET /state     → Get current state
  - GET /schema    → Action/Observation schemas
"""
from __future__ import annotations

import os
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .env import StartupEnv
from .models import StartupAction, StartupObservation, StartupState

# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #

app = FastAPI(
    title="AI Startup Founder Simulator",
    description="An OpenEnv-compliant RL environment for business strategy",
    version="1.0.0",
)

# Global environment instance
_env: Optional[StartupEnv] = None


def get_env() -> StartupEnv:
    global _env
    if _env is None:
        _env = StartupEnv()
    return _env


# ------------------------------------------------------------------ #
# API Endpoints
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return ROOT_HTML


@app.get("/health")
async def health():
    return {"status": "healthy", "env": "ai-startup-simulator", "version": "1.0.0"}


@app.post("/reset")
async def reset_env(req: ResetRequest = ResetRequest()):
    env = get_env()
    obs = env.reset(seed=req.seed)
    return obs.model_dump()


@app.post("/step")
async def step_env(req: StepRequest):
    env = get_env()
    action = StartupAction(action=req.action)
    obs = env.step(action)
    return obs.model_dump()


@app.get("/state")
async def get_state():
    env = get_env()
    return env.state.model_dump()


@app.get("/schema")
async def get_schema():
    return {
        "action": StartupAction.model_json_schema(),
        "observation": StartupObservation.model_json_schema(),
        "state": StartupState.model_json_schema(),
    }


# ------------------------------------------------------------------ #
# Root HTML Page
# ------------------------------------------------------------------ #

ROOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Startup Founder Simulator</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#0a0a1a;color:#e0e0e0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.container{max-width:900px;padding:40px;text-align:center}
h1{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}
.subtitle{color:#a0a0c0;font-size:1.05rem;margin-bottom:32px}
.badge{display:inline-block;background:linear-gradient(135deg,#10b981,#059669);color:#fff;padding:6px 16px;border-radius:20px;font-size:0.85rem;font-weight:600;margin-bottom:32px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-bottom:32px}
.card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:20px;text-align:left;transition:all .3s}
.card:hover{border-color:rgba(99,102,241,0.4);transform:translateY(-2px)}
.card h3{font-size:0.95rem;font-weight:700;color:#a78bfa;margin-bottom:6px}
.card p{font-size:0.84rem;color:#9090b0;line-height:1.5}
.actions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:32px}
.action{background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);color:#a78bfa;padding:5px 12px;border-radius:8px;font-size:0.8rem;font-family:'Inter',monospace}
.endpoints{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:20px;text-align:left}
.endpoints h3{color:#ec4899;margin-bottom:10px;font-size:1rem}
.ep{display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:0.88rem}
.method{background:#6366f1;color:#fff;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;min-width:48px;text-align:center}
.method.post{background:#f59e0b}
.path{color:#c0c0e0;font-family:monospace}
.desc{color:#808098;font-size:0.8rem}
.footer{margin-top:32px;color:#606080;font-size:0.8rem}
</style>
</head>
<body>
<div class="container">
<h1>&#x1F680; AI Startup Founder Simulator</h1>
<p class="subtitle">OpenEnv-compliant RL environment for AI agents learning real-world business strategy</p>
<span class="badge">&#x2705; Running &amp; OpenEnv Compliant</span>

<div class="grid">
<div class="card"><h3>&#x1F3AF; Easy: Runway</h3><p>Survive for 12 months without going bankrupt. Score = months / 12.</p></div>
<div class="card"><h3>&#x1F4C8; Medium: Market Fit</h3><p>Achieve &#x20B9;100,000 in monthly revenue at any point.</p></div>
<div class="card"><h3>&#x1F984; Hard: Unicorn</h3><p>Reach &#x20B9;1M valuation while keeping tech debt low.</p></div>
</div>

<h3 style="color:#a78bfa;margin-bottom:10px;font-size:0.95rem">Available Actions (8)</h3>
<div class="actions">
<span class="action">hire_engineer</span><span class="action">fire_employee</span>
<span class="action">build_feature</span><span class="action">run_marketing</span>
<span class="action">raise_funding</span><span class="action">do_nothing</span>
<span class="action">pivot</span><span class="action">train_team</span>
</div>

<div class="endpoints">
<h3>&#x1F50C; API Endpoints</h3>
<div class="ep"><span class="method post">POST</span><span class="path">/reset</span><span class="desc">- Reset environment</span></div>
<div class="ep"><span class="method post">POST</span><span class="path">/step</span><span class="desc">- Take an action</span></div>
<div class="ep"><span class="method">GET</span><span class="path">/state</span><span class="desc">- Current state</span></div>
<div class="ep"><span class="method">GET</span><span class="path">/health</span><span class="desc">- Health check</span></div>
<div class="ep"><span class="method">GET</span><span class="path">/schema</span><span class="desc">- API schemas</span></div>
</div>

<p class="footer">Meta PyTorch Hackathon &bull; OpenEnv Spec v1.0</p>
</div>
</body>
</html>"""


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    """CLI entry point (pyproject.toml [project.scripts])."""
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
