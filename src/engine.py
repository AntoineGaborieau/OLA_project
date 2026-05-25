import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

class Simulation:
    def __init__(self, env, agent, horizon, budget, values, conflict_edges):
        self.env = env
        self.agent = agent
        self.T = horizon
        self.B = budget
        self.v = values
        self.conflict_edges = conflict_edges
        
        self.rewards = np.zeros(self.T)
        self.costs = np.zeros(self.T)
        
    def run(self):
        for t in range(self.T):
            if self.agent.budget_remaining <= 0:
                break
                
            b_t = self.agent.bid()
            
            won_mask, utilities, costs = self.env.resolve_auctions(b_t)
            
            self.agent.update(won_mask, utilities, costs)
            
            self.rewards[t] = np.sum(utilities)
            self.costs[t] = np.sum(costs)
            
        clairvoyant_reward = self.compute_clairvoyant_optimal()
        agent_cumulative_reward = np.sum(self.rewards)
        terminal_regret = clairvoyant_reward - agent_cumulative_reward
        
        return self.rewards, self.costs, terminal_regret
        
    def compute_clairvoyant_optimal(self):
        m_seq = self.env.get_m_sequence()
        N = self.v.shape[0]
        num_vars = self.T * N
        
        utilities = self.v[np.newaxis, :] - m_seq
        utilities[utilities < 0] = 0
        
        c = -utilities.flatten()
        
        A_budget = m_seq.flatten().reshape(1, -1)
        b_u_budget = np.array([self.B])
        
        num_conflicts = self.T * len(self.conflict_edges)
        if num_conflicts > 0:
            A_conflict = np.zeros((num_conflicts, num_vars))
            row_idx = 0
            for t in range(self.T):
                for i, j in self.conflict_edges:
                    A_conflict[row_idx, t*N + i] = 1
                    A_conflict[row_idx, t*N + j] = 1
                    row_idx += 1
                    
            b_u_conflict = np.ones(num_conflicts)
            
            A = np.vstack([A_budget, A_conflict])
            b_u = np.concatenate([b_u_budget, b_u_conflict])
        else:
            A = A_budget
            b_u = b_u_budget
            
        b_l = np.full_like(b_u, -np.inf)
        
        constraints = LinearConstraint(A, b_l, b_u)
        bounds = Bounds(0, 1)
        integrality = np.ones_like(c)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if not res.success:
            raise ValueError("MILP solver failed to compute clairvoyant optimal.")
            
        return -res.fun
