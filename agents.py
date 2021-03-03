import numpy as np
import random

##################################################
### AGENTS pick actions and learn from rewards ###
##################################################

class Agent:
    def __init__(self, **parameters):
        raise NotImplementedError
    def select_action(self):
        ''' chooses an action based on previous experiences '''
        raise NotImplementedError
    def update(self, action, reward_signal):
        ''' updates internal variables based on most recent experience '''
        raise NotImplementedError

class FrustrationModelAgent(Agent):
    def __init__(self, reward, cost, expectation_growth, expectation_decay, temperature=0):
        self.policy = np.zeros(2, dtype=np.float32) # preference of each available action
        self.expectation = np.zeros(2, dtype=np.float32) # subtracted from both reward and punishment

        self.reward = reward # amount that the policy changes per reward received
        self.cost = cost # amount that the policy decreases per non-rewarded action
        self.expectation_growth = expectation_growth # growth of expectation for each rewarded movement
        self.expectation_decay = expectation_decay # decay of expectation for each non-rewarded movement
        self.exponent = np.e**(-temperature)

    def select_action(self):
        den = self.policy[0]**self.exponent + self.policy[1]**self.exponent
        if den == 0:
            prob_action_0 = .5
        else:
            prob_action_0 = self.policy[0]**self.exponent / den
        if random.random() < prob_action_0**self.exponent:
            return 0
        else:
            return 1

    def update(self, action, reward_signal):
        if reward_signal > 0:
            self.policy[action] += self.reward
            self.policy[action] -= self.expectation[action]
            self.expectation[action] += self.expectation_growth
            if self.policy[action] < 0: self.policy[action] = 0
        else:
            self.policy[action] -= self.cost
            self.policy[action] += self.expectation[action]
            self.expectation[action] -= self.expectation_decay
            if self.expectation[action] < 0: self.expectation[action] = 0
            if self.policy[action] < 0: self.policy[action] = 0

