import numpy as np

class SingleCampaignEnv:
    def __init__(self, budgets, sigma=2):
        self.budgets = np.array(budgets)
        self.sigma = sigma
        # unknown true reward curve
        self.means = 30 * (1 - np.exp(-self.budgets / 25))

    def round(self, pulled_arm):
        mean = self.means[pulled_arm]
        reward = np.random.normal(mean, self.sigma)
        return max(reward, 0)
