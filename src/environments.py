import numpy as np

class AuctionEnvironment:
    def __init__(self, values, T, distributions, conflict_matrix=None):
        self.v = np.array(values)
        self.N = len(self.v)
        self.T = T
        self.t = 0
        
        if conflict_matrix is None:
            self.conflict_matrix = np.zeros((self.N, self.N))
        else:
            self.conflict_matrix = np.array(conflict_matrix)
            np.fill_diagonal(self.conflict_matrix, 0)
            
        self.m_sequence = self._generate_m_sequence(distributions)
        
    def _generate_m_sequence(self, distributions):
        m_seq = np.zeros((self.T, self.N))
        for i, dist in enumerate(distributions):
            m_seq[:, i] = dist(size=self.T)
        return m_seq
        
    def get_m_sequence(self):
        return self.m_sequence
        
    def resolve_auctions(self, b_t):
        if self.t >= self.T:
            raise RuntimeError("Horizon T exceeded.")
            
        b_t = np.array(b_t)
        
        active_bids = (b_t > 0).astype(int)
        if np.dot(active_bids, np.dot(self.conflict_matrix, active_bids)) > 0:
            raise ValueError("Bid vector violates conflict graph constraints.")
            
        m_t = self.m_sequence[self.t]
        
        won_mask = b_t >= m_t
        costs = np.where(won_mask, b_t, 0)
        utilities = np.where(won_mask, self.v - costs, 0)
        
        self.t += 1
        
        return won_mask, utilities, costs
