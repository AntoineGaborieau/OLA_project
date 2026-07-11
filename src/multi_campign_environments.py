import numpy as np


class MultiCampaignEnvironment:
    def __init__(self, values, B, T, joint_distribution_func, conflict_matrix, cdf_functions):
        self.values = np.array(values, dtype=float)
        self.N = len(self.values)
        self.conflict_matrix = np.array(conflict_matrix, dtype=bool)
        self.cdf_functions = cdf_functions

        self.T = int(T)
        self.B = float(B)   
        self.t = 0
        self.spent = 0.0

        self.m_sequence = joint_distribution_func(size=self.T)

    def resolve_auction(self, bid_vector):
        if self.t >= self.T:
            raise RuntimeError("Horizon T exceeded.")

        bid_vector = np.array(bid_vector, dtype=float)

        # Strict combinatorial conflict check
        active_campaigns = np.where(bid_vector > 0.0)[0]
        for i in range(len(active_campaigns)):
            for j in range(i + 1, len(active_campaigns)):
                if self.conflict_matrix[active_campaigns[i], active_campaigns[j]]:
                    raise ValueError(f"Conflict: Campaigns {active_campaigns[i]} and {active_campaigns[j]}")

        # Theoretical expectation (for regret tracking), using the TRUE cdfs
        win_probs = np.zeros(self.N)
        for i in range(self.N):
            win_probs[i] = self.cdf_functions[i](bid_vector[i])
        expected_net = float(np.sum((self.values - bid_vector) * win_probs))

        # Physical market resolution
        m_t = self.m_sequence[self.t]
        won_vector = bid_vector >= m_t

        costs = np.where(won_vector, bid_vector, 0.0)
        rewards = np.where(won_vector, self.values - bid_vector, 0.0)  # strict net utility

        self.spent += np.sum(costs)
        self.t += 1

        return won_vector, rewards, costs, {"expected_net": expected_net}





# ============================================================================
# Non-stationary distribution factory (Requirement 3)
# ----------------------------------------------------------------------------
# Returns a joint_distribution_func(size=T) -> (T, N) array of highest competing
# bids. Each campaign's Beta mean follows an independent fast random walk with
# occasional abrupt jumps: "highly" non-stationary (no fixed switch points,
# distinct trajectory per campaign), as opposed to the piecewise-stationary
# Req 4 environment.
# ============================================================================

def make_nonstationary_beta_dist(N, T, seed=None):
    """
    N : int        number of campaigns
    T : int        horizon (only used for shape; drift keys off no fixed t)
    seed : int     RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    kappa = 8.0                                   # Beta concentration (obs noise)

    # Heterogeneous per-campaign drift speeds and jump rates, fixed for the run.
    step = rng.uniform(0.02, 0.06, size=N)        # random-walk step std per campaign
    jump_prob = rng.uniform(0.003, 0.01, size=N)  # per-round jump probability

    def joint_distribution_func(size):
        m = np.empty((size, N), dtype=float)
        mu = rng.uniform(0.2, 0.8, size=N)        # random starting mean per campaign

        for t in range(size):
            for i in range(N):
                # (a) abrupt regime jump at a random round
                if rng.random() < jump_prob[i]:
                    mu[i] = rng.uniform(0.1, 0.9)
                # (b) continuous fast drift, reflected at the [0.05, 0.95] edges
                else:
                    mu[i] += rng.normal(0.0, step[i])
                    if mu[i] < 0.05:
                        mu[i] = 0.10 - mu[i]
                    elif mu[i] > 0.95:
                        mu[i] = 1.90 - mu[i]

                a = mu[i] * kappa
                b = (1.0 - mu[i]) * kappa
                m[t, i] = rng.beta(a, b)

        return m

    return joint_distribution_func


# ============================================================================
# Full-feedback environment (Requirement 3)
# ----------------------------------------------------------------------------
# Subclasses the real budget-aware MultiCampaignEnvironment. The only change:
# resolve_auction additionally exposes the realized m_t vector (the highest
# competing bid for EVERY campaign), since Req 3 assumes FULL FEEDBACK. To
# avoid breaking the parent's 4-tuple contract, m_t is added into the existing
# info dict rather than appended as a 5th return value.
# ============================================================================

class FullFeedbackMultiCampaignEnvironment(MultiCampaignEnvironment):
    def resolve_auction(self, bid_vector):
        # m_t for the CURRENT round, captured before the parent advances self.t.
        # Guard against the horizon check so we read a valid index.
        m_t = self.m_sequence[self.t].copy() if self.t < self.T else None

        won_vector, rewards, costs, info = super().resolve_auction(bid_vector)

        # Full feedback: expose the entire competing-bid vector via info.
        info["m_t"] = m_t
        return won_vector, rewards, costs, info

    def observe_m(self, t=None):
        """Highest competing bids at round t (defaults to current round)."""
        idx = self.t if t is None else t
        return self.m_sequence[idx].copy()