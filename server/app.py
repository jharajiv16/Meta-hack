"""
AI Startup Founder Simulator — Server entry point.

Uses the OpenEnv create_app() factory to expose:
  - WebSocket-based step()/reset()/state() for agents
  - Optional Gradio web UI at /web (when ENABLE_WEB_INTERFACE=true)
  - Health check at /health
"""
from __future__ import annotations

import os

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
    return app


app = create_startup_app()


def main():
    """CLI entry point (pyproject.toml [project.scripts])."""
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
