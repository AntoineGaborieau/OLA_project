import numpy as np
from scipy import optimize


class MultiCampaignBaseLearner:
    """Multi-campaign analogue of SingleCampaignBaseLearner: N campaigns,
    each with its own valuation and bid space, sharing a single budget B
    over horizon T."""
    def __init__(self, values, bid_space, B, T):
        self.values = np.array(values, dtype=float)
        self.N = len(self.values)

        bid_space = np.array(bid_space, dtype=float)
        if bid_space.ndim == 1:
            self.bid_spaces = [bid_space.copy() for _ in range(self.N)]
        else:
            self.bid_spaces = [np.array(b, dtype=float) for b in bid_space]
        self.K = np.array([len(b) for b in self.bid_spaces])

        self.budget = float(B)
        self.T_horizon = float(T)
        self.spent = 0.0
        self.t = 0

        self.last_action_idx = None  # array of shape (N,), per-campaign bid index

    def pull_arm(self):
        return self.bid()

    def bid(self):
        raise NotImplementedError

    def update(self, won, utility, cost, m_t=None):
        raise NotImplementedError


class HedgeAgent:
    """Standard Hedge (Exponential Weights), full-feedback. Same convention
    as the reference notebook: update() takes a LOSS l_t over all arms
    (every arm's outcome is observed thanks to full feedback), and the
    agent minimizes cumulative loss."""
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K) / K
        self.a_t = None
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / np.sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate * l_t)
        self.t += 1


class PrimalDualMultiCampaignAgent(MultiCampaignBaseLearner):
    """
    Primal-dual bidding strategy with a single shared budget across N
    campaigns (Requirement 3).

    Primal: one HedgeAgent per campaign -- "a primal regret minimizer for
    the specific problem under study" (the hint), since full feedback
    (observing m_t for every campaign every round) makes a full-information
    regret minimizer like Hedge the natural choice; no exploration is needed.

    Dual: a single shared multiplier lambda_t, updated via online gradient
    ascent/descent using the TOTAL cost across all campaigns in the round,
    since the budget B is one pool shared by every campaign.
    """

    def __init__(self, values, bid_space, B, T, eta=None):
        super().__init__(values, bid_space, B, T)

        hedge_lr = np.sqrt(np.log(np.max(self.K)) / self.T_horizon)
        self.hedges = [HedgeAgent(K_i, hedge_lr) for K_i in self.K]

        self.rho = self.budget / self.T_horizon
        self.eta = eta if eta is not None else 1.0 / np.sqrt(self.T_horizon)
        self.lmbd = 1.0
        self.lmbd_max = 1.0 / self.rho

        # Aggregate bookkeeping in the same style as your single-campaign
        # learners (pulls / average_rewards), kept per campaign here.
        self.N_pulls = [np.zeros(K_i) for K_i in self.K]
        self.avg_f = [np.zeros(K_i) for K_i in self.K]
        self.avg_c = [np.zeros(K_i) for K_i in self.K]

    def bid(self):
        if self.budget < 1e-8:
            self.last_action_idx = np.zeros(self.N, dtype=int)
            return self.last_action_idx

        idx = np.empty(self.N, dtype=int)
        for i in range(self.N):
            idx[i] = self.hedges[i].pull_arm()
        self.last_action_idx = idx
        return self.last_action_idx

    def update(self, won, utility, cost, m_t):
        """
        won, utility, cost : array-like (N,)
            Realized outcome for each campaign this round (as returned by
            the environment's resolve_auction).
        m_t : array-like (N,)
            Highest competing bid for EVERY campaign this round (full
            feedback), regardless of which campaigns were actually won.
        """
        utility = np.array(utility, dtype=float)
        cost = np.array(cost, dtype=float)
        m_t = np.array(m_t, dtype=float)

        # Once the budget is exhausted, bid() always returns zero bids and
        # there is nothing left to learn or pace: freeze both the primal
        # (Hedge) and dual (lambda) updates so lambda does not drift back
        # down for the remainder of the horizon.
        if self.budget < 1e-8:
            self.t += 1
            return

        # --- primal update: one Hedge per campaign, using full feedback ---
        for i in range(self.N):
            b = self.bid_spaces[i]
            v_i = self.values[i]
            idx_i = self.last_action_idx[i]

            win_full = (b >= m_t[i]).astype(float)
            f_full = (v_i - b) * win_full
            c_full = b * win_full

            # Lagrangian per bid. The constant -lambda*rho offset is dropped:
            # it's identical across every arm in a campaign/round, so it
            # cancels out of Hedge's normalized weights after exponentiation
            # and renormalization -- it would only matter for the rescaling
            # constants below, which we compute directly from campaign-
            # specific bounds instead.
            L = f_full - self.lmbd * c_full

            b_max = b.max() if b.max() > 0 else 1.0
            L_up = v_i
            L_low = -self.lmbd_max * b_max
            if L_up > L_low:
                rescaled_L = (L - L_low) / (L_up - L_low)
            else:
                rescaled_L = np.zeros_like(L)

            self.hedges[i].update(1 - rescaled_L)  # Hedge minimizes -> maximize L

            # bookkeeping, mirroring your single-campaign learners' style
            self.N_pulls[i][idx_i] += 1
            n = self.N_pulls[i][idx_i]
            self.avg_f[i][idx_i] += (utility[i] - self.avg_f[i][idx_i]) / n
            self.avg_c[i][idx_i] += (cost[i] - self.avg_c[i][idx_i]) / n

        # --- dual update: single shared multiplier, total cost across campaigns ---
        total_cost = cost.sum()
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - total_cost),
                             a_min=0.0, a_max=self.lmbd_max)

        # --- budget bookkeeping ---
        self.spent += total_cost
        self.budget -= total_cost
        self.t += 1
