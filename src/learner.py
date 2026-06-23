import numpy as np

class SingleCampaignBaseLearner:
    def __init__(self, value, bid_space):
        self.v = float(value)
        self.bid_space = np.array(bid_space, dtype=float)
        self.K = len(self.bid_space)
        self.t = 0
        
        self.pulls = np.zeros(self.K, dtype=float)
        self.average_rewards = np.zeros(self.K, dtype=float)
        self.last_action_idx = None

    def pull_arm(self):
        return self.bid()

    def bid(self):
        raise NotImplementedError

    def update(self, won, utility, cost):
        self.pulls[self.last_action_idx] += 1
        n = self.pulls[self.last_action_idx]
        self.average_rewards[self.last_action_idx] += (utility - self.average_rewards[self.last_action_idx]) / n
        self.t += 1

class SingleCampaignUCB1Learner(SingleCampaignBaseLearner):
    def bid(self):
        if self.t < self.K:
            self.last_action_idx = self.t
        else:
            exploration_radius = np.sqrt(2 * np.log(np.maximum(self.t, 1.0)) / self.pulls)
            ucb_indices = self.average_rewards + exploration_radius
            self.last_action_idx = np.argmax(ucb_indices)
            
        return self.last_action_idx
