import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import random

class StartupEnv(gym.Env):
    """
    OpenAI Gym-style environment for simulated AI startup management.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path="openenv.yaml"):
        super(StartupEnv, self).__init__()

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)["environment"]

        # Action space: 0: hire_engineer, 1: fire_employee, 2: build_feature, 3: run_marketing_campaign, 4: raise_funding, 5: do_nothing
        self.action_space = spaces.Discrete(6)

        # Observation space
        self.observation_space = spaces.Dict({
            "cash": spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            "team_size": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            "product_quality": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "marketing_reach": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "revenue": spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            "burn_rate": spaces.Box(low=0, high=1e9, shape=(1,), dtype=np.float32),
            "month": spaces.Box(low=0, high=120, shape=(1,), dtype=np.int32)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.month = 0
        self.cash = self.config["initial_cash"]
        self.team_size = 1 # Founder
        self.product_quality = 0.1
        self.marketing_reach = 0.1
        self.revenue = 0
        self.burn_rate = self.config["burn_rate_per_employee"]
        self.done = False
        self.info = {"events": []}

        return self.get_state(), {}

    def get_state(self):
        return {
            "cash": np.array([self.cash], dtype=np.float32),
            "team_size": np.array([self.team_size], dtype=np.int32),
            "product_quality": np.array([self.product_quality], dtype=np.float32),
            "marketing_reach": np.array([self.marketing_reach], dtype=np.float32),
            "revenue": np.array([self.revenue], dtype=np.float32),
            "burn_rate": np.array([self.burn_rate], dtype=np.float32),
            "month": np.array([self.month], dtype=np.int32)
        }

    def step(self, action):
        self.month += 1
        self.info["events"] = []

        # Action logic
        if action == 0: # hire_engineer
            if self.cash >= self.config["hiring_cost"]:
                self.cash -= self.config["hiring_cost"]
                self.team_size += 1
                self.product_quality = min(1.0, self.product_quality + 0.05)
                self.info["events"].append("Hired an engineer")
        elif action == 1: # fire_employee
            if self.team_size > 1:
                self.team_size -= 1
                self.info["events"].append("Fired an employee")
        elif action == 2: # build_feature
            if self.cash >= self.config["feature_cost"]:
                self.cash -= self.config["feature_cost"]
                self.product_quality = min(1.0, self.product_quality + 0.1)
                self.info["events"].append("Built a feature")
        elif action == 3: # run_marketing_campaign
            if self.cash >= self.config["marketing_cost"]:
                self.cash -= self.config["marketing_cost"]
                self.marketing_reach = min(1.0, self.marketing_reach + 0.15)
                self.info["events"].append("Marketing campaign successful")
        elif action == 4: # raise_funding
            if random.random() < self.config["funding_probability"]:
                funding_amount = (self.revenue * 12) + (self.product_quality * 50000)
                funding_amount = max(10000, funding_amount)
                self.cash += funding_amount
                self.info["events"].append(f"Raised ₹{funding_amount:,.0f} in funding!")
            else:
                self.info["events"].append("Funding pitch failed")
        elif action == 5: # do_nothing
            self.info["events"].append("Idle month")

        # Update Burn Rate
        self.burn_rate = self.team_size * self.config["burn_rate_per_employee"]
        self.cash -= self.burn_rate

        # Update Revenue
        # Revenue is driven by product quality and marketing reach
        self.revenue = (self.product_quality * self.marketing_reach * 10000) + (self.team_size * 100)
        
        # Random Events
        if random.random() < self.config["random_event_probability"]:
            event_roll = random.random()
            if event_roll < 0.5: # Market crash
                self.revenue *= 0.5
                self.info["events"].append("MARKET CRASH! Revenue slashed.")
            else: # Viral growth
                self.marketing_reach = min(1.0, self.marketing_reach + 0.3)
                self.revenue *= 2.0
                self.info["events"].append("VIRAL GROWTH! Marketing exploded.")

        self.cash += self.revenue

        # Calculate Reward
        reward = self._calculate_reward()

        # Termination conditions
        terminated = False
        truncated = False

        if self.cash <= 0:
            terminated = True
            reward -= 5000 # Bankruptcy penalty
            self.info["events"].append("BANKRUPTCY. Game over.")
        
        if self.month >= self.config["max_months"]:
            truncated = True

        return self.get_state(), reward, terminated, truncated, self.info

    def _calculate_reward(self):
        # Weighted reward function
        reward = 0
        reward += (self.revenue / 1000) * 10
        reward += (self.product_quality * 50)
        reward += (self.marketing_reach * 30)
        reward -= (self.burn_rate / 1000) * 2
        reward += 10 # Survival bonus
        
        return reward

    def render(self):
        print(f"Month: {self.month} | Cash: ₹{self.cash:,.0f} | Team: {self.team_size} | Revenue: ₹{self.revenue:,.0f}")
