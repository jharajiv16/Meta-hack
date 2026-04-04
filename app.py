import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import StartupEnv
from inference import RuleBasedAgent, PPOAgentWrapper
from tasks import get_tasks
import io

class SimulatorUI:
    def __init__(self):
        self.env = StartupEnv()
        self.agent_rb = RuleBasedAgent()
        self.agent_ppo = PPOAgentWrapper()
        
        self.reset_sim()

    def reset_sim(self):
        self.obs, self.info = self.env.reset()
        self.history = []
        self._record_history(0, 0, [])
        return self._get_display_data()

    def _record_history(self, action, reward, events):
        self.history.append({
            "Month": int(self.obs["month"][0]),
            "Action": action,
            "Reward": float(reward),
            "Cash": float(self.obs["cash"][0]),
            "Team": int(self.obs["team_size"][0]),
            "Quality": float(self.obs["product_quality"][0]),
            "Reach": float(self.obs["marketing_reach"][0]),
            "Revenue": float(self.obs["revenue"][0]),
            "Events": ", ".join(events)
        })

    def _get_display_data(self):
        df = pd.DataFrame(self.history)
        
        # State display
        state_info = f"""
        **Month:** {int(self.obs["month"][0])} | **Cash:** ₹{float(self.obs["cash"][0]):,.0f} | **Team:** {int(self.obs["team_size"][0])} | **Revenue:** ₹{float(self.obs["revenue"][0]):,.0f} | **Quality:** {float(self.obs["product_quality"][0])*100:.1f}% | **Marketing Reach:** {float(self.obs["marketing_reach"][0])*100:.1f}%
        """
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(df) > 1:
            ax.plot(df["Month"], df["Cash"], label="Cash", color="green")
            ax.plot(df["Month"], df["Revenue"], label="Revenue", color="blue")
            ax.set_xlabel("Month")
            ax.set_ylabel("Currency")
            ax.legend()
            ax.set_title("Cash and Revenue Over Time")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        
        logs = df[["Month", "Action", "Revenue", "Cash", "Events"]].tail(5).to_html()
        
        return state_info, buf.getvalue(), logs

    def step_manual(self, action_idx):
        action_map = {
            "Hire Engineer": 0,
            "Fire Employee": 1,
            "Build Feature": 2,
            "Run Marketing": 3,
            "Raise Funding": 4,
            "Do Nothing": 5
        }
        action = action_map.get(action_idx, 5)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self._record_history(action_idx, reward, self.info["events"])
        
        data = self._get_display_data()
        if terminated:
            return "**BANKRUPTCY! Game Over.**", data[1], data[2]
        return data

    def run_agent(self, agent_type):
        agent = self.agent_rb if agent_type == "Rule-based" else self.agent_ppo
        
        while True:
            action = agent.get_action(self.obs)
            action_name = ["Hire Engineer", "Fire Employee", "Build Feature", "Run Marketing", "Raise Funding", "Do Nothing"][action]
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            self._record_history(action_name, reward, self.info["events"])
            
            if terminated or truncated:
                break
        
        return self._get_display_data()

sim = SimulatorUI()

with gr.Blocks(title="AI Startup Founder Simulator") as demo:
    gr.Markdown("# 🚀 AI Startup Founder Simulator")
    gr.Markdown("Control your startup, making decisions on hiring, product, and marketing to survive and thrive!")
    
    with gr.Row():
        state_display = gr.Markdown(value="**Initializing Simulator...**")
    
    with gr.Row():
        with gr.Column(scale=2):
            main_plot = gr.Image()
        with gr.Column(scale=1):
            with gr.Tab("Manual Decisions"):
                action_input = gr.Radio(
                    ["Hire Engineer", "Fire Employee", "Build Feature", "Run Marketing", "Raise Funding", "Do Nothing"],
                    label="Choose Action"
                )
                btn_step = gr.Button("Step Simulation", variant="primary")
            with gr.Tab("Automated Agent"):
                agent_select = gr.Radio(["Rule-based", "PPO (Skeleton)"], label="Choose Agent")
                btn_agent = gr.Button("Run Full Simulation", variant="secondary")
            
            btn_reset = gr.Button("Reset Simulator", variant="stop")

    with gr.Row():
        event_logs = gr.HTML(label="Recent Logs")

    btn_step.click(sim.step_manual, inputs=[action_input], outputs=[state_display, main_plot, event_logs])
    btn_agent.click(sim.run_agent, inputs=[agent_select], outputs=[state_display, main_plot, event_logs])
    btn_reset.click(sim.reset_sim, outputs=[state_display, main_plot, event_logs])

if __name__ == "__main__":
    demo.launch(share=True)
