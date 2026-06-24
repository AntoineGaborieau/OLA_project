import numpy as np
from scipy import optimize

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

    def update(self, won, reward, cost):
        self.pulls[self.last_action_idx] += 1
        n = self.pulls[self.last_action_idx]
        self.average_rewards[self.last_action_idx] += (reward - self.average_rewards[self.last_action_idx]) / n
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

class SmartUCB1Agent(SingleCampaignUCB1Learner):
    def bid(self):
        # Optimized initialization: Only pull physically uninitialized arms
        uninitialized_indices = np.where(self.pulls == 0)[0]
        if len(uninitialized_indices) > 0:
            self.last_action_idx = uninitialized_indices[0]
        else:
            exploration_radius = np.sqrt(2 * np.log(np.maximum(self.t, 1.0)) / self.pulls)
            ucb_indices = self.average_rewards + exploration_radius
            self.last_action_idx = np.argmax(ucb_indices)
            
        return self.last_action_idx

    def update(self, won, utility, cost):
        if won:
            inferred_indices = np.arange(self.last_action_idx, self.K)
            inferred_utilities = self.v - self.bid_space[inferred_indices]
        else:
            inferred_indices = np.arange(0, self.last_action_idx + 1)
            inferred_utilities = 0.0
            
        self.pulls[inferred_indices] += 1
        
        self.average_rewards[inferred_indices] += (
            (inferred_utilities - self.average_rewards[inferred_indices]) / self.pulls[inferred_indices]
        )
        self.t += 1

class BudgetConstrainedUCBAgent(SingleCampaignBaseLearner):
    def __init__(self, value, bid_space, B, T):
        super().__init__(value, bid_space)
        self.T_horizon = float(T)
        
        self.a_t = None
        self.avg_f = np.zeros(self.K)
        self.avg_c = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        
        self.budget = float(B)
        self.rho = self.budget / self.T_horizon
        self.t = 0

    def pull_arm(self):
        if self.budget < 1.0:
            self.a_t = 0 # Force selection of bid 0.0 to accumulate zero cost
            self.last_action_idx = self.a_t
            return self.a_t
            
        if self.t < self.K:
            self.a_t = self.t
        else:
            radius = np.sqrt(2 * np.log(self.t) / self.N_pulls)
            
            f_ucbs = self.avg_f + (self.v * radius)
            c_lcbs = self.avg_c - (1.0 * radius)
            
            gamma_t = self.compute_opt(f_ucbs, c_lcbs)
            self.a_t = np.random.choice(self.K, p=gamma_t)
            
        self.last_action_idx = self.a_t
        return self.a_t

    def compute_opt(self, f_ucbs, c_lcbs):
        if np.sum(c_lcbs <= np.zeros(len(c_lcbs))):
            gamma = np.zeros(len(f_ucbs))
            gamma[np.argmax(f_ucbs)] = 1
            return gamma
        c = -f_ucbs
        A_ub = [c_lcbs]
        b_ub = [self.rho]
        A_eq = [np.ones(self.K)]
        b_eq = [1]
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
        gamma = res.x
        return gamma

    def update(self, won, utility, cost):
        self.N_pulls[self.a_t] += 1
        
        self.avg_f[self.a_t] += (utility - self.avg_f[self.a_t]) / self.N_pulls[self.a_t]
        self.avg_c[self.a_t] += (cost - self.avg_c[self.a_t]) / self.N_pulls[self.a_t]
        
        self.budget -= cost
        
        self.pulls = self.N_pulls
        self.average_rewards = self.avg_f
        self.t += 1
