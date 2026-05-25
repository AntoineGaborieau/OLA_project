import numpy as np

class BaseEnvironment:
    def __init__(self, T):
        self.T = T
        self.t = 0

    def resolve_auction(self, b_t):
        raise NotImplementedError

    def get_m_sequence(self):
        raise NotImplementedError

class SingleCampaignEnvironment(BaseEnvironment):
    def __init__(self, value, T, distribution_func):
        super().__init__(T)
        self.v = float(value)
        # Pre-generate the entire sequence of highest competing bids for clairvoyant baseline
        self.m_sequence = distribution_func(size=self.T)

    def get_m_sequence(self):
        # Reshape to (T, 1) to match the general multi-campaign dimensions later
        return self.m_sequence.reshape(-1, 1)

    def resolve_auction(self, b_t):
        if self.t >= self.T:
            raise RuntimeError("Horizon T exceeded.")
        
        m_t = self.m_sequence[self.t]
        won = float(b_t >= m_t)
        
        cost = float(b_t) if won else 0.0
        utility = float(self.v - b_t) if won else 0.0
        
        self.t += 1
        return won, utility, cost
