import numpy as np
from environment import MultiCampaignEnvironment
from learners import MultiCampaignBudgetUCB1

T = 500
values = [1.0, 0.8, 1.2]          # value of each campaign
budget = 80
bid_space = np.linspace(0, 1, 11)  # possible bids

rng = np.random.default_rng(0)

distribution_funcs = [
    lambda size: rng.uniform(0.0, 1.0, size=size),
    lambda size: rng.beta(2, 5, size=size),
    lambda size: rng.beta(5, 2, size=size),
]

env = MultiCampaignEnvironment(
    values=values,
    T=T,
    distribution_funcs=distribution_funcs
)

agent = MultiCampaignBudgetUCB1(
    values=values,
    bid_space=bid_space,
    budget=budget
)

rewards = np.zeros(T)
costs = np.zeros(T)
chosen_campaigns = np.zeros(T)
chosen_bids = np.zeros(T)

for t in range(T):
    action = agent.pull_arm()
    campaign_idx, bid = action

    campaign_idx, won, utility, cost = env.resolve_auction(action)
    agent.update(won, utility, cost)

    rewards[t] = utility
    costs[t] = cost
    chosen_campaigns[t] = campaign_idx
    chosen_bids[t] = bid

print("Total reward:", np.sum(rewards))
print("Total cost:", np.sum(costs))
print("Remaining budget:", agent.budget_remaining)
print("Number of times each campaign-bid pair was selected:")
print(agent.pulls)
