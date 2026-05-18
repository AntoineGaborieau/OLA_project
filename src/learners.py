import numpy as np

class UCBLearner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.full(n_arms, np.inf)
        self.n_pulls = np.zeros(n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.argmax(upper_conf)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        n = self.n_pulls[pulled_arm]
        self.empirical_means[pulled_arm] = ((self.empirical_means[pulled_arm] * (n - 1) + reward) / n)
        for arm in range(self.n_arms):
            if self.n_pulls[arm] > 0:
                self.confidence[arm] = np.sqrt(2 * np.log(self.t) / self.n_pulls[arm])
