import numpy as np

class MultiCampaignBaseLearner:
    def __init__(self, values, bid_space):
        self.values = np.array(values, dtype=float)
        self.bid_space = np.array(bid_space, dtype=float)

        self.n_campaigns = len(self.values)
        self.K = len(self.bid_space)
        self.t = 0

        self.pulls = np.zeros((self.n_campaigns, self.K))
        self.wins = np.zeros((self.n_campaigns, self.K))

        self.last_campaign_idx = None
        self.last_bid_idx = None

    def pull_arm(self):
        return self.bid()

    def bid(self):
        raise NotImplementedError

    def update(self, won, utility, cost):
        self.pulls[self.last_campaign_idx, self.last_bid_idx] += 1
        self.wins[self.last_campaign_idx, self.last_bid_idx] += float(won)
        self.t += 1

class MultiCampaignUCB1(MultiCampaignBaseLearner):
    def bid(self):
        total_actions = self.n_campaigns * self.K

        # first try every campaign-bid pair once
        if self.t < total_actions:
            action_idx = self.t
            campaign_idx = action_idx // self.K
            bid_idx = action_idx % self.K
        else:
            p_hat = self.wins / self.pulls
            confidence = np.sqrt(2 * np.log(float(self.t)) / self.pulls)
            ucb = np.minimum(p_hat + confidence, 1.0)

            # expected utility = (campaign value - bid) * optimistic winning probability
            expected_utility = (self.values[:, None] - self.bid_space[None, :]) * ucb

            campaign_idx, bid_idx = np.unravel_index(
                np.argmax(expected_utility),
                expected_utility.shape
            )

        self.last_campaign_idx = campaign_idx
        self.last_bid_idx = bid_idx
        return campaign_idx, self.bid_space[bid_idx]

class MultiCampaignBudgetUCB1(MultiCampaignUCB1):
    def __init__(self, values, bid_space, budget):
        super().__init__(values, bid_space)
        self.budget_remaining = float(budget)

    def bid(self):
        total_actions = self.n_campaigns * self.K
        affordable = self.bid_space <= self.budget_remaining

        if not np.any(affordable):
            self.last_campaign_idx = 0
            self.last_bid_idx = 0
            return 0, 0.0

        if self.t < total_actions:
            action_idx = self.t
            campaign_idx = action_idx // self.K
            bid_idx = action_idx % self.K

            if not affordable[bid_idx]:
                campaign_idx, bid_idx = self._ucb_choice(affordable)
        else:
            campaign_idx, bid_idx = self._ucb_choice(affordable)

        self.last_campaign_idx = campaign_idx
        self.last_bid_idx = bid_idx
        return campaign_idx, self.bid_space[bid_idx]

    def _ucb_choice(self, affordable):
        safe_pulls = np.where(self.pulls > 0, self.pulls, 1.0)
        p_hat = np.where(self.pulls > 0, self.wins / safe_pulls, 1.0)
        confidence = np.where(
            self.pulls > 0,
            np.sqrt(2 * np.log(float(max(self.t, 1))) / safe_pulls),
            1.0
        )
        ucb = np.minimum(p_hat + confidence, 1.0)

        expected_utility = (self.values[:, None] - self.bid_space[None, :]) * ucb
        expected_utility[:, ~affordable] = -1e9

        campaign_idx, bid_idx = np.unravel_index(
            np.argmax(expected_utility),
            expected_utility.shape
        )
        return campaign_idx, bid_idx

    def update(self, won, utility, cost):
        super().update(won, utility, cost)
        self.budget_remaining -= float(cost)
