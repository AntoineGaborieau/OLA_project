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
        



class HeuristicBudgetUCBAgent(SingleCampaignBaseLearner):
    """
    UCB1-style bidding agent with a heuristic budget constraint for a single
    campaign (Requirement 1: "extend UCB1 to handle the budget constraint").

    Each round, paces spending by comparing every arm's current average
    observed cost against budget_per_round = remaining_budget / remaining_rounds
    (an even split of what's left over the rounds left), masks out arms whose
    average cost exceeds that pace, and plays the highest-UCB arm among the
    rest. Once remaining budget drops below 1 (or the horizon is reached),
    it stops bidding for the rest of the run.
    """
    def __init__(self, value, bid_space, B, T):
        super().__init__(value, bid_space)
        self.budget = float(B)
        self.T_horizon = float(T)
        self.a_t = None
        self.average_costs = np.zeros(self.K)
        self.spent = 0.0

    def pull_arm(self):
        remaining_budget = self.budget - self.spent
        remaining_rounds = self.T_horizon - self.t

        if remaining_budget <1 or remaining_rounds <= 0:
            self.a_t = 0
            self.last_action_idx = self.a_t
            return self.a_t

        if self.t < self.K:
            self.a_t = self.t
            self.last_action_idx = self.a_t
            return self.a_t

        budget_per_round = remaining_budget / remaining_rounds
        ucbs = self.average_rewards + np.sqrt(2 * np.log(self.t) / self.pulls)
        
        affordable = self.average_costs <= budget_per_round

        if not np.any(affordable):
            self.a_t = 0
        else:
            masked_ucbs = np.where(affordable, ucbs, -np.inf)
            self.a_t = np.argmax(masked_ucbs)

        self.last_action_idx = self.a_t
        return self.a_t

    def update(self, won, utility, cost):
        self.pulls[self.a_t] += 1
        n = self.pulls[self.a_t]
        
        self.average_rewards[self.a_t] += (utility - self.average_rewards[self.a_t]) / n
        self.average_costs[self.a_t] += (cost - self.average_costs[self.a_t]) / n
        
        self.spent += cost
        self.t += 1


class AvgCostBudgetUCBAgent(SingleCampaignBaseLearner):
    """
    Same LP-based structure as BudgetConstrainedUCBAgent, EXCEPT the cost side
    of the LP constraint uses the plain empirical average cost instead of a lower-confidence-bound.

    Reward side is unchanged: still an optimistic UCB (avg_f + v*radius)

    """

    def __init__(self, value, bid_space, B, T):
        super().__init__(value, bid_space)
        self.T_horizon = float(T)
        self.budget = float(B)

        self.a_t = None
        self.avg_f = np.zeros(self.K)
        self.avg_c = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        self.spent = 0.0

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
            self.last_action_idx = self.a_t
            return self.a_t
        if self.budget < 1:
            self.a_t = 0  
            self.last_action_idx = self.a_t
            return self.a_t
            
        radius = np.sqrt(2 * np.log(self.t) / self.N_pulls)

        f_ucbs = self.avg_f + (self.v * radius)
        # plug-in average cost, no confidence adjustment
        c_hat = self.avg_c

        gamma_t = self.compute_opt(f_ucbs, c_hat)
        self.a_t = np.random.choice(self.K, p=gamma_t)

        self.last_action_idx = self.a_t
        return self.a_t

    def compute_opt(self, f_ucbs, c_hat):
        remaining_budget = self.budget - self.spent
        remaining_rounds = np.maximum(self.T_horizon - self.t, 1.0)

        rho_t = remaining_budget / remaining_rounds

        c = -f_ucbs
        A_ub = [c_hat]
        b_ub = [rho_t]
        A_eq = [np.ones(self.K)]
        b_eq = [1.0]

        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs-ds')

        if res.success:
            gamma = np.clip(res.x, 0, 1)
            gamma /= np.sum(gamma)
            return gamma
        else:
            gamma = np.zeros(self.K)
            gamma[np.argmin(c_hat)] = 1.0
            return gamma

    def update(self, won, utility, cost):
        self.N_pulls[self.a_t] += 1
        n = self.N_pulls[self.a_t]

        self.avg_f[self.a_t] += (utility - self.avg_f[self.a_t]) / n
        self.avg_c[self.a_t] += (cost - self.avg_c[self.a_t]) / n

        self.spent += cost
        self.pulls = self.N_pulls
        self.average_rewards = self.avg_f
        self.t += 1


class BudgetConstrainedUCBAgent(SingleCampaignBaseLearner):
    """
    LP-based UCB agent for single-campaign bidding with a budget constraint.

    """

    def __init__(self, value, bid_space, B, T):
        super().__init__(value, bid_space)
        self.T_horizon = float(T)
        self.budget = float(B)

        self.a_t = None
        self.avg_f = np.zeros(self.K)
        self.avg_c = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        self.spent = 0.0

    def pull_arm(self):
        remaining_budget = self.budget - self.spent
        remaining_rounds = np.maximum(self.T_horizon - self.t, 1.0)

        if remaining_budget <= 1:
            self.a_t = 0
            self.last_action_idx = self.a_t
            return self.a_t

        if self.t < self.K:
            self.a_t = self.t
            self.last_action_idx = self.a_t
            return self.a_t

        radius = np.sqrt(2 * np.log(np.maximum(self.t, 1)) / self.N_pulls)  # fix 1: log(t)

        f_ucbs = self.avg_f + (self.v * radius)
        c_lcbs = self.avg_c - (self.bid_space * radius)

        gamma_t = self.compute_opt(f_ucbs, c_lcbs, remaining_budget, remaining_rounds)
        self.a_t = np.random.choice(self.K, p=gamma_t)

        self.last_action_idx = self.a_t
        return self.a_t

    def compute_opt(self, f_ucbs, c_lcbs, remaining_budget, remaining_rounds):
        rho_t = remaining_budget / remaining_rounds

        c = -f_ucbs
        A_ub = [c_lcbs]
        b_ub = [rho_t]
        A_eq = [np.ones(self.K)]
        b_eq = [1.0]

        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs-ds')

        if res.success:
            gamma = np.clip(res.x, 0, 1)
            gamma /= np.sum(gamma)
            return gamma
        else:
            gamma = np.zeros(self.K)
            gamma[np.argmin(c_lcbs)] = 1.0
            return gamma

    def update(self, won, utility, cost):
        self.N_pulls[self.a_t] += 1
        n = self.N_pulls[self.a_t]

        self.avg_f[self.a_t] += (utility - self.avg_f[self.a_t]) / n
        self.avg_c[self.a_t] += (cost - self.avg_c[self.a_t]) / n

        self.spent += cost
        self.pulls = self.N_pulls
        self.average_rewards = self.avg_f
        self.t += 1


