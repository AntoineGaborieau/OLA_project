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






##############################################################################################################
##############################  		Req 3 learners   #############################################
##############################################################################################################




class MultiCampaignBaseLearner:
    """N campaigns, each with its own valuation and bid space, sharing one
    budget B over horizon T. pull_arm() returns BID VALUES (shape (N,)), the
    same contract as the Req 2 C_UCB agents, so a single simulation loop can
    drive either agent."""

    def __init__(self, values, bid_space, B, T, conflict_matrix=None):
        self.values = np.array(values, dtype=float)
        self.N = len(self.values)

        bid_space = np.array(bid_space, dtype=float)
        if bid_space.ndim == 1:
            self.bid_spaces = [bid_space.copy() for _ in range(self.N)]
        else:
            self.bid_spaces = [np.array(b, dtype=float) for b in bid_space]
        self.K = np.array([len(b) for b in self.bid_spaces])

        if conflict_matrix is None:
            self.conflict_matrix = np.zeros((self.N, self.N), dtype=bool)
        else:
            self.conflict_matrix = np.array(conflict_matrix, dtype=bool)

        self.budget = float(B)
        self.T_horizon = float(T)
        self.spent = 0.0
        self.t = 0
        self.last_action_idx = np.zeros(self.N, dtype=int)

    def pull_arm(self):
        raise NotImplementedError

    def update(self, won, utility, cost, m_t=None):
        raise NotImplementedError

    def _bids_from_idx(self, idx):
        """Map per-campaign arm indices -> bid values."""
        return np.array([self.bid_spaces[i][idx[i]] for i in range(self.N)])

    def _resolve_conflicts(self, idx):
        """If two conflicting campaigns both bid (arm > 0), keep the higher-
        valuation one and force the other to abstain (arm 0). Same effect as
        the C_UCB agents' valid-super-arm restriction, enforced after the fact."""
        for i in range(self.N):
            if idx[i] == 0:
                continue
            for j in range(i + 1, self.N):
                if idx[j] != 0 and self.conflict_matrix[i, j]:
                    drop = j if self.values[i] >= self.values[j] else i
                    idx[drop] = 0
        return idx


class HedgeAgent:
    """Hedge / Exponential Weights, full feedback. update() takes a LOSS
    vector over all arms and minimizes cumulative loss."""

    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)

    def pull_arm(self):
        x = self.weights / self.weights.sum()
        return np.random.choice(self.K, p=x)

    def update(self, loss):
        self.weights *= np.exp(-self.learning_rate * loss)


class PrimalDualMultiCampaignAgent(MultiCampaignBaseLearner):
    """Primal-dual bidding, one shared budget across N campaigns (Req 3).

    Primal: one Hedge per campaign (full feedback -> no exploration needed).
    Dual:   one shared multiplier lambda, updated by online gradient descent
            on the total per-round cost, since the budget is one shared pool.
    """

    def __init__(self, values, bid_space, B, T, conflict_matrix=None, eta=None):
        super().__init__(values, bid_space, B, T, conflict_matrix)

        hedge_lr = np.sqrt(np.log(self.K.max()) / self.T_horizon)
        self.hedges = [HedgeAgent(K_i, hedge_lr) for K_i in self.K]

        self.rho = self.budget / self.T_horizon          # per-round budget target
        self.eta = eta if eta is not None else 1.0 / np.sqrt(self.T_horizon)
        self.lmbd = 1.0
        self.lmbd_max = 1.0 / self.rho

    def pull_arm(self):
        # No budget left: abstain on every campaign.
        if self.budget < 1e-8:
            self.last_action_idx = np.zeros(self.N, dtype=int)
            return self._bids_from_idx(self.last_action_idx)

        idx = np.array([h.pull_arm() for h in self.hedges], dtype=int)
        idx = self._resolve_conflicts(idx)
        self.last_action_idx = idx
        return self._bids_from_idx(idx)

    def update(self, won, utility, cost, m_t):
        cost = np.array(cost, dtype=float)
        m_t = np.array(m_t, dtype=float)

        # Budget exhausted: freeze primal and dual so lambda doesn't drift back.
        if self.budget < 1e-8:
            self.t += 1
            return

        # --- primal: one Hedge per campaign, full feedback on m_t ---
        for i in range(self.N):
            b = self.bid_spaces[i]
            v_i = self.values[i]

            win = (b >= m_t[i]).astype(float)        # would each bid have won?
            f = (v_i - b) * win                      # utility per bid
            c = b * win                              # cost per bid
            L = f - self.lmbd * c                    # Lagrangian per bid

            # Rescale L into [0, 1] using campaign-specific bounds, then feed
            # Hedge a loss (1 - L) so minimizing loss maximizes L.
            L_up = v_i
            L_low = -self.lmbd_max * b.max()
            rescaled = (L - L_low) / (L_up - L_low) if L_up > L_low else np.zeros_like(L)
            self.hedges[i].update(1.0 - rescaled)

        # --- dual: single shared multiplier on total cost ---
        total_cost = cost.sum()
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - total_cost),
                            0.0, self.lmbd_max)

        # --- budget bookkeeping ---
        self.spent += total_cost
        self.budget -= total_cost
        self.t += 1

class DiscountedHedgeAgent:
    """Discounted Hedge / Fixed-Share Exponential Weights, full feedback.

    Identical to HedgeAgent except the accumulated log-weights are DISCOUNTED
    by a forgetting factor gamma in (0, 1] before each update. This makes the
    distribution track a MOVING optimum instead of converging to a single fixed
    one: old evidence decays, so the agent can re-weight toward whatever is best
    *now* when the market drifts.

    gamma = 1.0 recovers plain Hedge (no forgetting). gamma slightly below 1
    (e.g. 0.999) gives an effective memory of ~1/(1-gamma) rounds.
    """

    def __init__(self, K, learning_rate, gamma):
        self.K = K
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Track log-weights rather than raw weights so the geometric discount is
        # a simple multiply and we avoid under/overflow over a long horizon.
        self.logw = np.zeros(K)

    def pull_arm(self):
        # Softmax over log-weights (shift by max for numerical stability).
        z = self.logw - self.logw.max()
        x = np.exp(z)
        x /= x.sum()
        return np.random.choice(self.K, p=x)

    def update(self, loss):
        # (1) DISCOUNT: shrink all accumulated evidence toward 0 (=uniform).
        self.logw *= self.gamma
        # (2) Standard multiplicative-weights step, in log space.
        self.logw -= self.learning_rate * loss


class NSPrimalDualMultiCampaignAgent(PrimalDualMultiCampaignAgent):
    def __init__(self, values, bid_space, B, T, conflict_matrix=None, eta=None,
                 gamma=0.999):
        super().__init__(values, bid_space, B, T, conflict_matrix, eta)
        self.gamma = gamma
        # Tune the learning rate to the EFFECTIVE memory window, not the full
        # horizon. With forgetting, only ~1/(1-gamma) recent rounds carry weight,
        # so the lr must be sized to that window or the primal is too sluggish
        # both to learn and to re-learn after drift (which destabilises the dual).
        eff_window = 1.0 / (1.0 - gamma) if gamma < 1.0 else self.T_horizon
        hedge_lr = np.sqrt(np.log(self.K.max()) / eff_window)
        self.hedges = [DiscountedHedgeAgent(K_i, hedge_lr, gamma) for K_i in self.K]