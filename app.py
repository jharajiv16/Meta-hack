from fastapi import FastAPI
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from env import StartupEnv
import io
from PIL import Image

# =========================
# ENV + SIMULATOR CLASS
# =========================
class SimulatorUI:
    def __init__(self):
        self.env = StartupEnv()
        self.reset_sim()

    def reset_sim(self):
        self.obs, self.info = self.env.reset()
        self.history = []
        return self._get_display()

    def _safe(self, x):
        try:
            return float(x[0])
        except:
            return float(x)

    def _get_display(self):
        try:
            cash = self._safe(self.obs["cash"])
            revenue = self._safe(self.obs["revenue"])
            month = int(self._safe(self.obs["month"]))
            team = int(self._safe(self.obs["team_size"]))

            # 📊 State text
            state = f"""
### 📊 Current State
- Month: {month}
- Cash: ₹{cash:,.0f}
- Team Size: {team}
- Revenue: ₹{revenue:,.0f}
"""

            # 📈 Plot
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(["Cash", "Revenue"], [cash, revenue])
            ax.set_title("Startup Metrics")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)

            img = Image.open(buf)

            return state, img, "<p>Simulation Running...</p>"

        except Exception as e:
            return f"❌ ERROR: {str(e)}", None, ""

    def step(self, action_name):
        try:
            if action_name is None:
                action_name = "Do Nothing"

            action_map = {
                "Hire Engineer": 0,
                "Fire Employee": 1,
                "Build Feature": 2,
                "Run Marketing": 3,
                "Raise Funding": 4,
                "Do Nothing": 5,
            }

            action = action_map[action_name]

            self.obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated:
                return "💀 BANKRUPT! Game Over", None, ""

            return self._get_display()

        except Exception as e:
            return f"❌ ERROR: {str(e)}", None, ""


# =========================
# INIT SIM
# =========================
sim = SimulatorUI()

# =========================
# FASTAPI (FOR HACKATHON CHECKS)
# =========================
app = FastAPI()

@app.post("/reset")
def reset_api():
    sim.reset_sim()
    return {"status": "reset done"}

@app.post("/step")
def step_api(action: int):
    try:
        action_map = {
            0: "Hire Engineer",
            1: "Fire Employee",
            2: "Build Feature",
            3: "Run Marketing",
            4: "Raise Funding",
            5: "Do Nothing",
        }

        sim.step(action_map.get(action, "Do Nothing"))

        return {"status": "step done"}

    except Exception as e:
        return {"error": str(e)}


# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 AI Startup Founder Simulator")

    state = gr.Markdown()
    plot = gr.Image()
    logs = gr.HTML()

    action = gr.Radio([
        "Hire Engineer",
        "Fire Employee",
        "Build Feature",
        "Run Marketing",
        "Raise Funding",
        "Do Nothing"
    ], label="Choose Action")

    step_btn = gr.Button("Step Simulation")
    reset_btn = gr.Button("Reset")

    step_btn.click(sim.step, inputs=action, outputs=[state, plot, logs])
    reset_btn.click(sim.reset_sim, outputs=[state, plot, logs])


# =========================
# MOUNT BOTH (VERY IMPORTANT)
# =========================
app = gr.mount_gradio_app(app, demo, path="/")