import os
import requests
import numpy as np
from stable_baselines3 import PPO

# ✅ REQUIRED ENV VARIABLES (LLM Support)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

def call_model(prompt):
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    url = f"{API_BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a startup advisor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def get_startup_advice(state):
    prompt = f"You are an expert startup mentor. Current startup state: {state}. Give 1 short actionable advice."
    return call_model(prompt)

# =========================
# REINFORCEMENT LEARNING AGENTS
# =========================

class RuleBasedAgent:
    """
    A simple rule-based agent for the StartupEnv.
    """
    def get_action(self, obs):
        cash = float(obs["cash"][0])
        team_size = int(obs["team_size"][0])
        quality = float(obs["product_quality"][0])
        reach = float(obs["marketing_reach"][0])
        morale = float(obs["team_morale"][0])

        # Priority 1: Survival
        if cash < 20000:
            return 4 # raise_funding
        
        # Priority 2: Quality/Product
        if quality < 0.5:
            if team_size < 3:
                return 0 # hire_engineer
            return 2 # build_feature
        
        # Priority 3: Growth
        if reach < 0.3:
            return 3 # run_marketing
            
        # Priority 4: Health
        if morale < 0.6:
            return 7 # train_team

        return 5 # do_nothing

class PPOAgentWrapper:
    """
    Wrapper for a PPO agent trained on StartupEnv.
    """
    def __init__(self, model_path=None):
        from env import StartupEnv
        self.env = StartupEnv()
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=self.env)
        else:
            # Fallback to a fresh skeleton model
            self.model = PPO("MultiInputPolicy", self.env, verbose=1)

    def get_action(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return int(action)