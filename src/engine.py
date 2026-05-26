import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

class SingleCampaignSimulation:
    def __init__(self, env, agent, horizon, budget, value):
        self.env = env
        self.agent = agent
        self.T = horizon
        self.B = float(budget)
        self.v = float(value)
        
        self.rewards = np.zeros(self.T)
        self.costs = np.zeros(self.T)
        
    def run(self):
        for t in range(self.T):
            # 1. Pull arm from Slide 14 agent interface
            b_t = self.agent.pull_arm()
            
            # 2. Resolve via single campaign environment
            won, utility, cost = self.env.resolve_auction(b_t)
            
            # 3. Inform the agent of the auction feedback
            self.agent.update(won, utility, cost)
            
            # 4. Record history
            self.rewards[t] = utility
            self.costs[t] = cost
            
        # Calculate regret relative to the exact offline Knapsack optimal
        clairvoyant_reward = self.compute_clairvoyant_optimal()
        agent_cumulative_reward = np.sum(self.rewards)
        terminal_regret = clairvoyant_reward - agent_cumulative_reward
        
        return self.rewards, self.costs, terminal_regret
        
    def compute_clairvoyant_optimal(self):
        m_seq = self.env.get_m_sequence().flatten()
        
        # Net utility for winning each specific round
        utilities = self.v - m_seq
        utilities[utilities < 0] = 0.0
        
        c = -utilities
        A = m_seq.reshape(1, -1)
        b_u = np.array([self.B])
        b_l = np.array([-np.inf])
        
        constraints = LinearConstraint(A, b_l, b_u)
        bounds = Bounds(0, 1)
        integrality = np.ones_like(c)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if not res.success:
            raise ValueError("MILP solver failed to compute clairvoyant optimal.")
            
        return -res.fun
