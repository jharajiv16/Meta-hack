from env import StartupEnv
import numpy as np

class Task:
    def __init__(self, name, description, eval_fn):
        self.name = name
        self.description = description
        self.eval_fn = eval_fn

    def evaluate(self, trajectory):
        """
        Evaluate a trajectory (list of observations).
        Returns a score between 0 and 1.
        """
        return self.eval_fn(trajectory)

def evaluate_easy(trajectory):
    """Easy: Survive 12 months."""
    months_survived = len(trajectory)
    score = min(1.0, months_survived / 12)
    return score

def evaluate_medium(trajectory):
    """Medium: Reach ₹100,000 revenue."""
    if not trajectory:
        return 0.0
    
    max_revenue = max([obs["revenue"][0] for obs in trajectory])
    score = min(1.0, max_revenue / 100000)
    return score

def evaluate_hard(trajectory):
    """Hard: Maximize valuation. valuation = revenue * 10 + product_quality * 50000."""
    if not trajectory:
        return 0.0
    
    valuate_fn = lambda obs: (obs["revenue"][0] * 10) + (obs["product_quality"][0] * 50000)
    max_valuation = max([valuate_fn(obs) for obs in trajectory])
    
    # Target valuation for hard task: ₹1,000,000
    score = min(1.0, max_valuation / 1000000)
    return score

def evaluate_sustainable(trajectory):
    """Sustainable Growth: Reach ₹500,000 valuation with low tech debt."""
    if not trajectory:
        return 0.0
    
    valuate_fn = lambda obs: (obs["revenue"][0] * 10) + (obs["product_quality"][0] * 50000)
    scores = []
    for obs in trajectory:
        valuation = valuate_fn(obs)
        tech_debt = obs["tech_debt"][0]
        # Penalty for high tech debt
        effective_valuation = valuation * (1.0 - tech_debt)
        scores.append(effective_valuation)
    
    max_score = max(scores)
    return min(1.0, max_score / 500000)

def evaluate_morale_leader(trajectory):
    """Morale Leader: Build a 10+ team with high morale."""
    if not trajectory:
        return 0.0
    
    scores = []
    for obs in trajectory:
        team_size = obs["team_size"][0]
        morale = obs["team_morale"][0]
        # Score based on team size (up to 10) and morale
        score = (min(10, team_size) / 10) * morale
        scores.append(score)
    
    return max(scores)

def get_tasks():
    return [
        Task("Easy", "Survive 12 months without going bankrupt.", evaluate_easy),
        Task("Medium", "Reach ₹100,000 in monthly revenue.", evaluate_medium),
        Task("Hard", "Reach a valuation of ₹1,000,000.", evaluate_hard),
        Task("Sustainable", "Reach ₹500,000 valuation while keeping tech debt low.", evaluate_sustainable),
        Task("Morale Leader", "Build a team of 10 with morale above 80%.", evaluate_morale_leader)
    ]
