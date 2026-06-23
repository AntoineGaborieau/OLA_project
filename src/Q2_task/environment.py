import numpy as np

class BaseEnvironment:
    def __init__(self, T):
        self.T = T
        self.t = 0

    def resolve_auction(self, action):
        raise NotImplementedError

    def get_m_sequence(self):
        raise NotImplementedError

class MultiCampaignEnvironment(BaseEnvironment):
    def __init__(self, values, T, distribution_funcs):
        super().__init__(T)
        self.values = np.array(values, dtype=float)
        self.n_campaigns = len(self.values)

        self.m_sequence = np.zeros((self.T, self.n_campaigns))
        for c in range(self.n_campaigns):
            self.m_sequence[:, c] = distribution_funcs[c](size=self.T)

    def get_m_sequence(self):
        return self.m_sequence

    def resolve_auction(self, action):
        if self.t >= self.T:
            raise RuntimeError("Horizon T exceeded.")

        campaign_idx, bid = action
        campaign_idx = int(campaign_idx)
        bid = float(bid)

        m_t = self.m_sequence[self.t, campaign_idx]

        won = float(bid >= m_t)

        cost = bid if won else 0.0
        utility = self.values[campaign_idx] - bid if won else 0.0
        self.t += 1
        return campaign_idx, won, utility, cost

