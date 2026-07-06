import numpy as np

# The real, budget-aware MultiCampaignEnvironment (Req 1/2) now lives in the
# shared module. Import it instead of the local placeholder that used to be here.
from multi_campign_environments import MultiCampaignEnvironment


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