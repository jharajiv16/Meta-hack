import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI
import gradio as gr
from env import StartupEnv
from inference import RuleBasedAgent, PPOAgentWrapper

# =========================
# SIMULATOR WRAPPER
# =========================
class SimulatorUI:
    def __init__(self):
        self.env = StartupEnv()
        self.agent_rb = RuleBasedAgent()
        self.agent_ppo = PPOAgentWrapper()
        self.reset_sim()

    def reset_sim(self):
        self.obs, self.info = self.env.reset()
        self.history = []
        self._record_history("Initial", 0, [])
        return self._get_display()

    def _safe(self, x):
        try: return float(x[0])
        except: return float(x)

    def _record_history(self, action, reward, events):
        self.history.append({
            "Month": int(self._safe(self.obs["month"])),
            "Action": action,
            "Reward": float(reward),
            "Cash": self._safe(self.obs["cash"]),
            "Revenue": self._safe(self.obs["revenue"]),
            "Quality": self._safe(self.obs["product_quality"]),
            "Reach": self._safe(self.obs["marketing_reach"]),
            "Morale": self._safe(self.obs["team_morale"]),
            "TechDebt": self._safe(self.obs["tech_debt"]),
            "Sentiment": self._safe(self.obs["market_sentiment"]),
            "Events": ", ".join(events) if events else "None"
        })

    def _get_display(self):
        df = pd.DataFrame(self.history)
        
        # 🧪 Core Stats for Markdown
        cash = self._safe(self.obs["cash"])
        rev = self._safe(self.obs["revenue"])
        month = int(self._safe(self.obs["month"]))
        team = int(self._safe(self.obs["team_size"]))
        morale = self._safe(self.obs["team_morale"])
        debt = self._safe(self.obs["tech_debt"])
        quality = self._safe(self.obs["product_quality"])
        reach = self._safe(self.obs["marketing_reach"])
        sentiment = self._safe(self.obs["market_sentiment"])

        state_info = f"""
**Month:** {month} | **Cash:** ₹{cash:,.0f} | **Team:** {team} | **Revenue:** ₹{rev:,.0f}
**Quality:** {quality*100:.1f}% | **Reach:** {reach*100:.1f}% | **Morale:** {morale*100:.1f}% | **Tech Debt:** {debt*100:.1f}%
**Market Sentiment:** {sentiment:.2f}x ({'Bullish' if sentiment > 1.1 else 'Bearish' if sentiment < 0.9 else 'Neutral'})
"""

        # 📈 Plot Generation
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor("#1e272e")
        fig.patch.set_facecolor("#1e272e")
        
        if len(df) > 1:
            ax.plot(df["Month"], df["Cash"], label="Cash Capital", color="#27ae60", linewidth=3)
            ax.plot(df["Month"], df["Revenue"], label="Monthly Revenue", color="#3498db", linewidth=2)
            ax.fill_between(df["Month"], df["Cash"], alpha=0.1, color="#27ae60")
            ax.set_title("Financial Trajectory", color="white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.legend(facecolor="#2c3e50", edgecolor="#34495e", labelcolor="white")
            ax.grid(color="#34495e", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Initializing Simulator...", ha='center', color='gray', fontsize=12)

        # 📜 Logs
        logs = df[["Month", "Action", "Revenue", "Events"]].tail(5).to_html(classes='mystyle', index=False)
        logs_html = f"<style>.mystyle {{width:100%; border-collapse:collapse; color:white;}} .mystyle td, .mystyle th {{padding:8px; border:1px solid #34495e; text-align:left;}}</style>{logs}"

        return state_info, fig, logs_html

    def step_manual(self, action_name):
        action_map = {
            "Hire Engineer": 0, "Fire Employee": 1, "Build Feature": 2, 
            "Run Marketing": 3, "Raise Funding": 4, "Do Nothing": 5, 
            "Pivot": 6, "Train Team": 7
        }
        action = action_map.get(action_name, 5)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self._record_history(action_name, reward, self.info.get("events", []))
        
        if terminated:
            return "**💀 BANKRUPTCY! GAME OVER.**", plt.figure(), "<h3>GAME OVER</h3>"
        
        return self._get_display()

    def run_agent(self, agent_type):
        agent = self.agent_rb if agent_type == "Rule-based" else self.agent_ppo
        while True:
            action = agent.get_action(self.obs)
            action_name = ["Hire Engineer", "Fire Employee", "Build Feature", "Run Marketing", "Raise Funding", "Do Nothing", "Pivot", "Train Team"][action]
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            self._record_history(f"🤖 {action_name}", reward, self.info.get("events", []))
            if terminated or truncated: break
        return self._get_display()

# =========================
# APP & UI INIT
# =========================
sim = SimulatorUI()
app = FastAPI()

# 🚀 API ENDPOINTS (REQUIRED FOR HACKATHON CHECKS)
@app.post("/reset")
def reset_api():
    sim.reset_sim()
    return {"status": "success", "message": "Environment reset"}

@app.post("/step")
def step_api(action: int):
    try:
        action_names = ["Hire Engineer", "Fire Employee", "Build Feature", "Run Marketing", "Raise Funding", "Do Nothing", "Pivot", "Train Team"]
        action_name = action_names[action] if 0 <= action < len(action_names) else "Do Nothing"
        sim.step_manual(action_name)
        return {"status": "success", "action": action_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate")) as demo:
    gr.Markdown("# 🚀 AI Startup Founder Simulator")
    gr.Markdown("Control your startup, making decisions on hiring, product, and marketing to survive and thrive!")

    with gr.Row():
        with gr.Column(scale=2):
            state_display = gr.Markdown(value="**Initializing Simulator...**")
            main_plot = gr.Plot()
            
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Manual Decisions"):
                    action_input = gr.Radio(
                        ["Hire Engineer", "Fire Employee", "Build Feature", "Run Marketing", "Raise Funding", "Do Nothing", "Pivot", "Train Team"],
                        label="Choose Action",
                        value="Do Nothing"
                    )
                    btn_step = gr.Button("Step Simulation", variant="primary")
                    btn_reset = gr.Button("Reset Simulator", variant="stop")
                
                with gr.Tab("Automated Agent"):
                    agent_select = gr.Radio(["Rule-based", "PPO (RL Model)"], label="Select Agent", value="Rule-based")
                    btn_agent = gr.Button("Run Full Simulation", variant="secondary")

    with gr.Row():
        event_logs = gr.HTML(label="Recent Logs")

    # Wire up events
    btn_step.click(sim.step_manual, inputs=action_input, outputs=[state_display, main_plot, event_logs])
    btn_agent.click(sim.run_agent, inputs=agent_select, outputs=[state_display, main_plot, event_logs])
    btn_reset.click(sim.reset_sim, outputs=[state_display, main_plot, event_logs])
    demo.load(sim.reset_sim, outputs=[state_display, main_plot, event_logs])

# =========================
# MOUNT & LAUNCH
# =========================
app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)