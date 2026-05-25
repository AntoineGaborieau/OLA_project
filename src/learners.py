import numpy as np

class SingleCampaignUCB1:
    def __init__(self, value, bid_space, budget):
        self.v = value
        self.bid_space = np.array(bid_space)
        self.K = len(self.bid_space)
        self.budget_remaining = budget
        self.t = 0
        
        self.pulls = np.zeros(self.K)
        self.wins = np.zeros(self.K)
        self.last_action_idx = None

    def bid(self):
        if self.t < self.K:
            action_idx = self.t
        else:
            p_hat = self.wins / self.pulls
            ucb = np.minimum(p_hat + np.sqrt(2 * np.log(self.t) / self.pulls), 1.0)
            expected_utility = (self.v - self.bid_space) * ucb
            action_idx = np.argmax(expected_utility)
            
        self.last_action_idx = action_idx
        return np.array([self.bid_space[action_idx]])

    def update(self, won_mask, utilities, costs):
        self.pulls[self.last_action_idx] += 1
        self.wins[self.last_action_idx] += won_mask[0]
        self.budget_remaining -= costs[0]
        self.t += 1


class SingleCampaignBudgetUCB1(SingleCampaignUCB1):
    def bid(self):
        affordable_mask = self.bid_space <= self.budget_remaining
        
        if not np.any(affordable_mask):
            return np.array([0.0]) # Budget depleted, bid zero

        if self.t < self.K and affordable_mask[self.t]:
            action_idx = self.t
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                p_hat = np.where(self.pulls > 0, self.wins / self.pulls, 1.0)
                exploration = np.where(self.pulls > 0, np.sqrt(2 * np.log(max(self.t, 1)) / self.pulls), 1.0)
                
            ucb = np.minimum(p_hat + exploration, 1.0)
            expected_utility = (self.v - self.bid_space) * ucb
            
            # Mask out mathematically impossible actions
            expected_utility[~affordable_mask] = -np.inf
            action_idx = np.argmax(expected_utility)
            
        self.last_action_idx = action_idx
        return np.array([self.bid_space[action_idx]])
