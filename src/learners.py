import numpy as np

class SingleCampaignBaseLearner:
    def __init__(self, value, bid_space):
        self.v = float(value)
        self.bid_space = np.array(bid_space, dtype=float)
        self.K = len(self.bid_space)
        self.t = 0
        
        self.pulls = np.zeros(self.K, dtype=float)
        self.wins = np.zeros(self.K, dtype=float)
        self.last_action_idx = None

    def bid(self):
        raise NotImplementedError

    def update(self, won, utility, cost):
        self.pulls[self.last_action_idx] += 1
        self.wins[self.last_action_idx] += float(won)
        self.t += 1

class SingleCampaignUCB1(SingleCampaignBaseLearner):
    """Algorithm 1: Standard UCB1 ignoring the budget constraint."""
    def bid(self):
        if self.t < self.K:
            action_idx = self.t
        else:
            p_hat = self.wins / self.pulls
            ucb = np.minimum(p_hat + np.sqrt(2 * np.log(self.t) / self.pulls), 1.0)
            expected_utility = (self.v - self.bid_space) * ucb
            action_idx = np.argmax(expected_utility)
            
        self.last_action_idx = action_idx
        return self.bid_space[action_idx]

class SingleCampaignBudgetUCB1(SingleCampaignUCB1):
    """Algorithm 2: Extended UCB1 handling the budget constraint."""
    def __init__(self, value, bid_space, budget):
        super().__init__(value, bid_space)
        self.budget_remaining = float(budget)

    def bid(self):
        affordable_mask = self.bid_space <= self.budget_remaining
        if not np.any(affordable_mask):
            return 0.0 # Force zero bid if budget is entirely depleted

        if self.t < self.K and affordable_mask[self.t]:
            action_idx = self.t
        else:
            safe_pulls = np.where(self.pulls > 0, self.pulls, 1.0)
            p_hat = np.where(self.pulls > 0, self.wins / safe_pulls, 1.0)
            exploration = np.where(self.pulls > 0, np.sqrt(2 * np.log(float(max(self.t, 1))) / safe_pulls), 1.0)
                
            ucb = np.minimum(p_hat + exploration, 1.0)
            expected_utility = (self.v - self.bid_space) * ucb
            expected_utility[~affordable_mask] = -1e9 # Mask unaffordable actions
            action_idx = np.argmax(expected_utility)
            
        self.last_action_idx = action_idx
        return self.bid_space[action_idx]

    def update(self, won, utility, cost):
        super().update(won, utility, cost)
        self.budget_remaining -= float(cost)
