import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env import StartupEnv

class RuleBasedAgent:
    """A rule-based agent that uses predefined business logic."""
    def __init__(self):
        pass

    def get_action(self, obs):
        cash = obs["cash"][0]
        team_size = obs["team_size"][0]
        quality = obs["product_quality"][0]
        reach = obs["marketing_reach"][0]
        revenue = obs["revenue"][0]

        # Prioritize survival and improvement
        if cash < 5000:
             # Critical cash: Try to raise funding or do nothing
             return 4 # raise_funding
        
        if quality < 0.4:
            # Low quality: Build feature
            return 2 # build_feature
        
        if team_size < 3 and cash > 15000:
             # Scale team: Hire engineer
             return 0 # hire_engineer
        
        if reach < 0.3:
            # Low reach: Run marketing
            return 3 # run_marketing_campaign
        
        if quality < 0.7:
             return 2 # build_feature
        
        if reach < 0.8:
            return 3 # run_marketing_campaign
        
        # Default action
        return 5 # do_nothing

class PPOAgentWrapper:
    """Wrapper for a PPO model from Stable-Baselines3."""
    def __init__(self, model_path=None):
        self.env = StartupEnv()
        if model_path:
            self.model = PPO.load(model_path)
        else:
            self.model = PPO("MultiInputPolicy", self.env, verbose=1)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def get_action(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action
