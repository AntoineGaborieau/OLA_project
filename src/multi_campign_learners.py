import numpy as np
import itertools
from scipy import optimize

class C_UCB_Bidding:
    """

    cutoff_mode controls how the hard "stop bidding" safety threshold is
    computed (the one that guarantees spent can never exceed budget):
      - 'refined_cutoff':      exact worst-case cost among the VALID
                                (conflict-respecting) super-arms. Tight.
      - 'conservative_cutoff': N * bid_space.max(). Ignores conflicts and
                                per-campaign value caps entirely -- a safe
                                but looser upper bound.
    """
    def __init__(self, values, bid_space, B, T, conflict_matrix, cutoff_mode='conservative_cutoff'):
        self.values = np.array(values, dtype=float)
        self.bid_space = np.array(bid_space, dtype=float)
        self.N, self.K = len(self.values), len(self.bid_space)
        self.T_horizon, self.budget = float(T), float(B)
        self.conflict_matrix = np.array(conflict_matrix, dtype=bool)

        self.avg_f = np.zeros((self.N, self.K))
        self.avg_c = np.zeros((self.N, self.K))
        self.N_pulls = np.zeros((self.N, self.K))

        self.spent = 0.0
        self.t = 0
        self.super_arms = self._generate_valid_super_arms()
        self.M = len(self.super_arms)
        self.last_action_indices = None
        self.last_sa_idx = None

        self.super_arm_worst_cost = np.sum(self.bid_space[self.super_arms], axis=1)
        self.cutoff_mode = cutoff_mode
        if cutoff_mode == 'refined_cutoff':
            self.max_possible_cost = self.super_arm_worst_cost.max()
        elif cutoff_mode == 'conservative_cutoff':
            self.max_possible_cost = self.N * self.bid_space.max()
        else:
            raise ValueError(f"Unknown cutoff_mode: {cutoff_mode!r} (use 'refined_cutoff' or 'conservative_cutoff')")

        self.zero_sa_idx = int(np.where(np.all(self.super_arms == 0, axis=1))[0][0])

    def _generate_valid_super_arms(self):
        valid_marginal = [[k for k, b in enumerate(self.bid_space) if b <= self.values[i]] for i in range(self.N)]
        valid_arms = []
        for combo in itertools.product(*valid_marginal):
            conflict = False
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if self.conflict_matrix[i][j] and combo[i] != 0 and combo[j] != 0:
                        conflict = True
                        break
                if conflict: break
            if not conflict: valid_arms.append(combo)
        return np.array(valid_arms, dtype=int)

    def _get_bounds(self):
        pulled = self.N_pulls > 0
        radius = np.zeros((self.N, self.K))
        safe_t = max(self.t, 1)
        radius[pulled] = np.sqrt(2.0 * np.log(safe_t + 1) / self.N_pulls[pulled])

        max_net = np.max(self.values[:, None] - self.bid_space[None, :], axis=1)
        f_ucbs = np.where(pulled, np.minimum(self.avg_f + max_net[:, None] * radius, max_net[:, None]), max_net[:, None])
        c_lcbs = np.where(pulled, np.maximum(self.avg_c - radius, 1e-8), 1e-8)

        f_ucbs[:, 0] = 0.0
        c_lcbs[:, 0] = 0.0
        return f_ucbs, c_lcbs

    def _safe_fallback(self):
        """Bid the all-zero super-arm: the only action with guaranteed 0 cost."""
        self.last_sa_idx = self.zero_sa_idx
        self.last_action_indices = self.super_arms[self.zero_sa_idx]
        return self.bid_space[self.last_action_indices]

    def update(self, won_vector, util_vector, cost_vector):
        self.spent += np.sum(cost_vector)
        i_idx = np.arange(self.N)
        k_idx = self.last_action_indices
        self.N_pulls[i_idx, k_idx] += 1
        n = self.N_pulls[i_idx, k_idx]
        self.avg_f[i_idx, k_idx] += (util_vector - self.avg_f[i_idx, k_idx]) / n
        self.avg_c[i_idx, k_idx] += (cost_vector - self.avg_c[i_idx, k_idx]) / n
        self.t += 1
class ARGMAX_agent(C_UCB_Bidding):
    """A greedy heuristic. Applies hard cutoffs based on LCB."""
    def pull_arm(self):
        rem_b, rem_r = self.budget - self.spent, max(self.T_horizon - self.t, 1.0)

        if rem_b < self.max_possible_cost or rem_r <= 0:
            return self._safe_fallback()

        f_ucbs, c_lcbs = self._get_bounds()
        U_sa = np.sum(f_ucbs[np.arange(self.N), self.super_arms], axis=1)
        C_sa = np.sum(c_lcbs[np.arange(self.N), self.super_arms], axis=1)

        valid_mask = C_sa <= (rem_b / rem_r)
        chosen_idx = np.argmax(np.where(valid_mask, U_sa, -np.inf)) if np.any(valid_mask) else np.argmin(C_sa)
        self.last_action_indices = self.super_arms[chosen_idx]
        return self.bid_space[self.last_action_indices]

class LP_agent(C_UCB_Bidding):
    def pull_arm(self):
        rem_b, rem_r = self.budget - self.spent, max(self.T_horizon - self.t, 1.0)

        if rem_b < self.max_possible_cost or rem_r <= 0:
            return self._safe_fallback()

        f_ucbs, c_lcbs = self._get_bounds()
        U_sa = np.sum(f_ucbs[np.arange(self.N), self.super_arms], axis=1)
        C_sa = np.sum(c_lcbs[np.arange(self.N), self.super_arms], axis=1) 

        res = optimize.linprog(
            -U_sa, A_ub=C_sa.reshape(1, -1), b_ub=[rem_b / rem_r],
            A_eq=np.ones((1, self.M)), b_eq=[1.0], bounds=[(0.0, 1.0)] * self.M, method='highs-ds'
        )
        gamma = np.clip(res.x, 0.0, 1.0) / np.clip(res.x, 0.0, 1.0).sum() if res.success else np.eye(self.M)[np.argmin(C_sa)]
        self.last_action_indices = self.super_arms[np.random.choice(self.M, p=gamma)]
        return self.bid_space[self.last_action_indices]

class LP_MeanCost_agent(C_UCB_Bidding):
    """Replaces LCB with empirical mean bounds because budget is depleted too fast with the optimistic bounds on cost"""
    def pull_arm(self):
        rem_b, rem_r = self.budget - self.spent, max(self.T_horizon - self.t, 1.0)

        if rem_b < self.max_possible_cost or rem_r <= 0:
            return self._safe_fallback()

        f_ucbs, _ = self._get_bounds()
        U_sa = np.sum(f_ucbs[np.arange(self.N), self.super_arms], axis=1)

        c_hats = np.where(self.N_pulls > 0, self.avg_c, 1e-8)
        c_hats[:, 0] = 0.0
        C_sa_hat = np.sum(c_hats[np.arange(self.N), self.super_arms], axis=1)

        res = optimize.linprog(
            -U_sa, A_ub=C_sa_hat.reshape(1, -1), b_ub=[rem_b / rem_r],
            A_eq=np.ones((1, self.M)), b_eq=[1.0], bounds=[(0.0, 1.0)] * self.M, method='highs-ds'
        )
        gamma = np.clip(res.x, 0.0, 1.0) / np.clip(res.x, 0.0, 1.0).sum() if res.success else np.eye(self.M)[np.argmin(C_sa_hat)]
        self.last_sa_idx = np.random.choice(self.M, p=gamma)
        self.last_action_indices = self.super_arms[self.last_sa_idx]
        return self.bid_space[self.last_action_indices]



class LP_MeanCost_RefinedCutoff(LP_MeanCost_agent):
    def __init__(self, values, bid_space, B, T, conflict_matrix):
        super().__init__(values, bid_space, B, T, conflict_matrix, cutoff_mode='refined_cutoff')

class LP_MeanCost_ConservativeCutoff(LP_MeanCost_agent):
    def __init__(self, values, bid_space, B, T, conflict_matrix):
        super().__init__(values, bid_space, B, T, conflict_matrix, cutoff_mode='conservative_cutoff')