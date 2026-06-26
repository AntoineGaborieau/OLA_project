import numpy as np

# ============================================================================
# Base classes (Requirement 1/2) -- bundled here so this file is self-contained
# and testable before the real MultiCampaignEnvironment is shared. When your
# friend's version lands, you can delete these two classes and import instead.
# ============================================================================

class BaseEnvironment:
    def __init__(self, T):
        self.T = T
        self.t = 0

    def resolve_auction(self, b_t):
        raise NotImplementedError

    def get_m_sequence(self):
        raise NotImplementedError


class MultiCampaignEnvironment(BaseEnvironment):
    def __init__(self, values, T, joint_distribution_func, conflict_matrix):
        super().__init__(T)
        self.values = np.array(values, dtype=float)
        self.N = len(self.values)
        self.conflict_matrix = np.array(conflict_matrix, dtype=bool)

        # joint_distribution_func must return an array of shape (T, N)
        self.m_sequence = joint_distribution_func(size=self.T)

    def get_m_sequence(self):
        return self.m_sequence

    def resolve_auction(self, bid_vector):
        if self.t >= self.T:
            raise RuntimeError("Horizon T exceeded.")

        bid_vector = np.array(bid_vector, dtype=float)

        if len(bid_vector) != self.N:
            raise ValueError(f"Bid vector dimension mismatch. Expected {self.N}, got {len(bid_vector)}.")

        # 1. Strict Conflict Enforcement (Environmental Firewall)
        active_campaigns = np.where(bid_vector > 0.0)[0]
        for i in range(len(active_campaigns)):
            for j in range(i + 1, len(active_campaigns)):
                if self.conflict_matrix[active_campaigns[i], active_campaigns[j]]:
                    raise ValueError(f"Combinatorial Violation: Campaigns {active_campaigns[i]} and {active_campaigns[j]} are mutually exclusive.")

        # 2. Market Resolution
        m_t = self.m_sequence[self.t]
        won_vector = bid_vector >= m_t

        # 3. Vectorized Metric Computation
        costs = np.where(won_vector, bid_vector, 0.0)
        rewards = np.where(won_vector, self.values - bid_vector, 0.0)

        self.t += 1
        return won_vector, rewards, costs


# ============================================================================
# Non-stationary distribution factory (Requirement 3)
# ----------------------------------------------------------------------------
# Produces a joint_distribution_func compatible with MultiCampaignEnvironment:
# it returns an array of shape (T, N). Here the Beta parameters of each
# campaign change over time, yielding a non-stationary sequence of highest
# competing bids.
# ============================================================================

def make_nonstationary_beta_dist(N, schedule, seed=None):
    """
    Build a joint_distribution_func(size=T) -> (T, N) array of highest
    competing bids, drawn from per-campaign Beta distributions whose
    parameters (a, b) vary over time.

    Parameters
    ----------
    N : int
        Number of campaigns.
    schedule : callable
        schedule(t, i) -> (a, b), the Beta parameters for campaign i at round t.
        This is where the non-stationarity lives.
    seed : int or None
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    def joint_distribution_func(size):
        T = size
        m = np.empty((T, N), dtype=float)
        for t in range(T):
            for i in range(N):
                a, b = schedule(t, i)
                m[t, i] = rng.beta(a, b)
        return m

    return joint_distribution_func





# ============================================================================
# Full-feedback environment (Requirement 3)
# ----------------------------------------------------------------------------
# Inherits from MultiCampaignEnvironment. The only change is that
# resolve_auction also returns the realized m_t vector, since Req 3 assumes
# FULL FEEDBACK: the agent observes the highest competing bid for every
# campaign, not just the ones it won.
# ============================================================================

class FullFeedbackMultiCampaignEnvironment(MultiCampaignEnvironment):
    def resolve_auction(self, bid_vector):
        # m_t for the CURRENT round, captured before the parent advances self.t
        m_t = self.m_sequence[self.t].copy()
        won_vector, rewards, costs = super().resolve_auction(bid_vector)
        # Full feedback: expose the entire competing-bid vector
        return won_vector, rewards, costs, m_t

    def observe_m(self, t=None):
        """Highest competing bids at round t (defaults to current round)."""
        idx = self.t if t is None else t
        return self.m_sequence[idx].copy()
