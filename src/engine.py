import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

class SingleCampaignSimulation:
    def __init__(self, env, agent, horizon, budget, value, bid_space):
        self.env = env
        self.agent = agent
        self.T = horizon
        self.B = float(budget)
        self.v = float(value)
        self.bid_space = np.array(bid_space, dtype=float)
        
        self.rewards = np.zeros(self.T)
        self.costs = np.zeros(self.T)
        
    def run(self):
        for t in range(self.T):
            b_t = self.agent.pull_arm()
            won, utility, cost = self.env.resolve_auction(b_t)
            self.agent.update(won, utility, cost)
            
            self.rewards[t] = utility
            self.costs[t] = cost
            
        # Extract the true step-by-step cumulative trajectory vector
        clairvoyant_trajectory = self.compute_clairvoyant_optimal()
        agent_cumulative_reward = np.cumsum(self.rewards)
        regret_curve = clairvoyant_trajectory - agent_cumulative_reward
        
        return self.rewards, self.costs, regret_curve
        
    def compute_clairvoyant_optimal(self):
        m_seq = self.env.get_m_sequence().flatten()
        K = len(self.bid_space)
        num_vars = self.T * K
        
        bids_matrix = self.bid_space[np.newaxis, :]
        m_matrix = m_seq[:, np.newaxis]
        
        won_matrix = (bids_matrix >= m_matrix).astype(float)
        utilities = (self.v - bids_matrix) * won_matrix
        costs = bids_matrix * won_matrix
        
        c = -utilities.flatten()
        
        # Constraint 1: Budget limit over the horizon
        A_budget = costs.flatten().reshape(1, -1)
        b_u_budget = np.array([self.B])
        
        # Constraint 2: Exactly ONE discrete bid choice per round t
        A_one_bid = np.zeros((self.T, num_vars))
        for t in range(self.T):
            A_one_bid[t, t*K : (t+1)*K] = 1.0
        b_u_one_bid = np.ones(self.T)
        
        A = np.vstack([A_budget, A_one_bid])
        b_u = np.concatenate([b_u_budget, b_u_one_bid])
        b_l = np.concatenate([[-np.inf], np.ones(self.T)])
        
        constraints = LinearConstraint(A, b_l, b_u)
        bounds = Bounds(0, 1)
        integrality = np.ones_like(c)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if not res.success:
            raise ValueError("MILP solver failed.")
            
        # Reshape solution binary mask back to shape (T, K)
        chosen_actions_mask = res.x.reshape(self.T, K)
        
        # Extract the exact utility obtained by the solver at each round t
        round_rewards = np.sum(utilities * chosen_actions_mask, axis=1)
        
        # Return the true chronological cumulative sum array
        return np.cumsum(round_rewards)
