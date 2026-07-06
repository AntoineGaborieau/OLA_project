import numpy as np


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