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