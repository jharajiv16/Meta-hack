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

        # Action space: 0: hire_engineer, 1: fire_employee, 2: build_feature, 3: run_marketing_campaign, 4: raise_funding, 5: do_nothing, 6: pivot, 7: train_team
        self.action_space = spaces.Discrete(8)

        # Observation space
        self.observation_space = spaces.Dict({
            "cash": spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            "team_size": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            "product_quality": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "marketing_reach": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "revenue": spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            "burn_rate": spaces.Box(low=0, high=1e9, shape=(1,), dtype=np.float32),
            "month": spaces.Box(low=0, high=120, shape=(1,), dtype=np.int32),
            "tech_debt": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "team_morale": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "market_sentiment": spaces.Box(low=0.5, high=1.5, shape=(1,), dtype=np.float32)
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
        self.tech_debt = 0.1
        self.team_morale = 1.0
        self.market_sentiment = 1.0
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
            "month": np.array([self.month], dtype=np.int32),
            "tech_debt": np.array([self.tech_debt], dtype=np.float32),
            "team_morale": np.array([self.team_morale], dtype=np.float32),
            "market_sentiment": np.array([self.market_sentiment], dtype=np.float32)
        }

    def step(self, action):
        self.month += 1
        self.info["events"] = []

        # Action logic
        if action == 0: # hire_engineer
            if self.cash >= self.config["hiring_cost"]:
                self.cash -= self.config["hiring_cost"]
                self.team_size += 1
                quality_gain = 0.05 * (1.0 - self.tech_debt) * self.team_morale
                self.product_quality = min(1.0, self.product_quality + quality_gain)
                self.tech_debt = min(1.0, self.tech_debt + self.config["tech_debt_increment_per_hire"])
                self.info["events"].append("Hired an engineer. Tech debt increased.")
        elif action == 1: # fire_employee
            if self.team_size > 1:
                self.team_size -= 1
                self.team_morale = max(0.0, self.team_morale - 0.2)
                self.info["events"].append("Fired an employee. Morale dropped.")
        elif action == 2: # build_feature
            if self.cash >= self.config["feature_cost"]:
                self.cash -= self.config["feature_cost"]
                quality_gain = 0.1 * (1.0 - self.tech_debt) * self.team_morale
                self.product_quality = min(1.0, self.product_quality + quality_gain)
                self.tech_debt = min(1.0, self.tech_debt + self.config["tech_debt_increment_per_feature"])
                self.info["events"].append("Built a feature. Tech debt increased.")
        elif action == 3: # run_marketing_campaign
            if self.cash >= self.config["marketing_cost"]:
                self.cash -= self.config["marketing_cost"]
                reach_gain = 0.15 * self.market_sentiment
                self.marketing_reach = min(1.0, self.marketing_reach + reach_gain)
                self.info["events"].append("Marketing campaign successful.")
        elif action == 4: # raise_funding
            prob = self.config["funding_probability"] * self.market_sentiment
            if random.random() < prob:
                funding_amount = (self.revenue * 15) + (self.product_quality * 100000)
                funding_amount = max(20000, funding_amount)
                self.cash += funding_amount
                self.info["events"].append(f"Raised ₹{funding_amount:,.0f} in funding!")
            else:
                self.info["events"].append("Funding pitch failed.")
        elif action == 5: # do_nothing
            self.tech_debt = max(0.0, self.tech_debt - 0.02)
            self.info["events"].append("Idle month. Tech debt slightly reduced.")
        elif action == 6: # pivot
            if self.cash >= self.config["pivot_cost"]:
                self.cash -= self.config["pivot_cost"]
                self.product_quality = 0.2
                self.tech_debt = 0.0
                self.info["events"].append("Pivoted! Quality reset, Tech Debt cleared.")
        elif action == 7: # train_team
            if self.cash >= self.config["training_cost"]:
                self.cash -= self.config["training_cost"]
                self.team_morale = min(1.0, self.team_morale + self.config["morale_recovery_per_training"])
                self.info["events"].append("Trained team. Morale boosted.")

        # Update Morale Decay
        self.team_morale = max(0.0, self.team_morale - self.config["morale_decay_per_month"])
        if self.cash < self.burn_rate * 2:
            self.team_morale = max(0.0, self.team_morale - 0.05)
            self.info["events"].append("Low runway is hurting team morale!")

        # Update Burn Rate
        self.burn_rate = self.team_size * self.config["burn_rate_per_employee"]
        self.cash -= self.burn_rate

        # Update Revenue
        # Revenue is driven by product quality and marketing reach
        self.revenue = (self.product_quality * self.marketing_reach * 10000) + (self.team_size * 100)
        
        # Random Events
        if random.random() < self.config["random_event_probability"]:
            event_roll = random.random()
            if event_roll < 0.3: # Market crash
                self.market_sentiment = max(0.5, self.market_sentiment - 0.3)
                self.revenue *= 0.5
                self.info["events"].append("MARKET CRASH! Sentiment plummeted.")
            elif event_roll < 0.6: # Viral growth
                self.marketing_reach = min(1.0, self.marketing_reach + 0.3)
                self.market_sentiment = min(1.5, self.market_sentiment + 0.2)
                self.info["events"].append("VIRAL GROWTH! Marketing exploded.")
            elif event_roll < 0.8: # Competitor launch
                self.marketing_reach = max(0.0, self.marketing_reach - 0.2)
                self.info["events"].append("COMPETITOR LAUNCH! Market share lost.")
            else: # Economic boom
                self.market_sentiment = min(1.5, self.market_sentiment + 0.3)
                self.info["events"].append("ECONOMIC BOOM! Investors are bullish.")

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
        reward += (self.revenue / 1000) * 15
        reward += (self.product_quality * 100)
        reward += (self.marketing_reach * 50)
        reward += (self.team_morale * 40)
        reward -= (self.tech_debt * 60)
        reward -= (self.burn_rate / 1000) * 5
        reward += 20 # Survival bonus
        
        return reward

    def render(self):
        print(f"Month: {self.month} | Cash: ₹{self.cash:,.0f} | Team: {self.team_size} | Revenue: ₹{self.revenue:,.0f}")
