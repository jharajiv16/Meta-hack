"""
AI Startup Founder Simulator — Server entry point.

Uses the OpenEnv create_app() factory to expose:
  - WebSocket-based step()/reset()/state() for agents
  - Health check at /health
  - Root page at / for HF Spaces iframe display
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core import create_app

from .env import StartupEnv
from .models import StartupAction, StartupObservation


def create_startup_app():
    """Factory: create the FastAPI app using the OpenEnv standard."""
    app = create_app(
        env=StartupEnv,
        action_cls=StartupAction,
        observation_cls=StartupObservation,
        env_name="ai-startup-simulator",
    )

    # Add a root route for the HF Spaces iframe
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return ROOT_HTML

    return app


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
h1{font-size:3rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}
.subtitle{color:#a0a0c0;font-size:1.1rem;margin-bottom:40px}
.badge{display:inline-block;background:linear-gradient(135deg,#10b981,#059669);color:#fff;padding:6px 16px;border-radius:20px;font-size:0.85rem;font-weight:600;margin-bottom:32px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;margin-bottom:40px}
.card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;text-align:left;transition:all .3s}
.card:hover{border-color:rgba(99,102,241,0.4);transform:translateY(-2px)}
.card h3{font-size:1rem;font-weight:700;color:#a78bfa;margin-bottom:8px}
.card p{font-size:0.88rem;color:#9090b0;line-height:1.5}
.actions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:36px}
.action{background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);color:#a78bfa;padding:6px 14px;border-radius:8px;font-size:0.82rem;font-family:'Inter',monospace}
.endpoints{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:24px;margin-top:20px;text-align:left}
.endpoints h3{color:#ec4899;margin-bottom:12px}
.ep{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:0.9rem}
.method{background:#6366f1;color:#fff;padding:2px 10px;border-radius:6px;font-size:0.78rem;font-weight:700;min-width:50px;text-align:center}
.method.ws{background:#10b981}
.path{color:#c0c0e0;font-family:monospace}
.desc{color:#808098;font-size:0.82rem}
.footer{margin-top:40px;color:#606080;font-size:0.82rem}
</style>
</head>
<body>
<div class="container">
<h1>🚀 AI Startup Founder Simulator</h1>
<p class="subtitle">An OpenEnv-compliant RL environment for training AI agents on real-world business strategy</p>
<span class="badge">✅ OpenEnv Compliant</span>

<div class="grid">
<div class="card">
<h3>🎯 Easy: Runway</h3>
<p>Survive for 12 months without going bankrupt. Score = months_survived / 12.</p>
</div>
<div class="card">
<h3>📈 Medium: Market Fit</h3>
<p>Achieve ₹100,000 in monthly revenue at any point. Score = best_revenue / 100k.</p>
</div>
<div class="card">
<h3>🦄 Hard: Unicorn</h3>
<p>Reach ₹1M valuation while keeping tech debt low. Penalised by debt level.</p>
</div>
</div>

<h3 style="color:#a78bfa;margin-bottom:12px">Available Actions (8)</h3>
<div class="actions">
<span class="action">hire_engineer</span>
<span class="action">fire_employee</span>
<span class="action">build_feature</span>
<span class="action">run_marketing</span>
<span class="action">raise_funding</span>
<span class="action">do_nothing</span>
<span class="action">pivot</span>
<span class="action">train_team</span>
</div>

<div class="endpoints">
<h3>🔌 API Endpoints</h3>
<div class="ep"><span class="method ws">WS</span><span class="path">/ws</span><span class="desc">— WebSocket for step/reset/state</span></div>
<div class="ep"><span class="method">GET</span><span class="path">/health</span><span class="desc">— Server health check</span></div>
<div class="ep"><span class="method">GET</span><span class="path">/schema</span><span class="desc">— Action & Observation schemas</span></div>
</div>

<p class="footer">Built for the Meta PyTorch Hackathon • OpenEnv Spec v1.0</p>
</div>
</body>
</html>"""


app = create_startup_app()


def main():
    """CLI entry point (pyproject.toml [project.scripts])."""
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
