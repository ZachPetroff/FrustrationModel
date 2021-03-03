import random

#####################################################
### ENVIRONMENTS take actions and provide rewards ###
###     (possibly in a time-dependent manner)     ###
#####################################################

class Environment:
    def __init__(self, **parameters):
        raise NotImplementedError
    def assign_reward(self, action):
        raise NotImplementedError

class FixedSlotMachineEnvironment(Environment):
    def __init__(self, prob_action_0):
        self.prob_action_0 = prob_action_0
    def assign_reward(self, action):
        reward_action_is_0 = random.random() < self.prob_action_0
        # get the reward if exactly one of (action == 1) and (reward_action == 0) is true
        if action ^ reward_action_is_0:
            return 1
        else:
            return 0

class SwitchingSlotMachineEnvironment(FixedSlotMachineEnvironment):
    def __init__(self, prob_action_0, switch_times):
        self.prob_action_0 = prob_action_0
        self.switch_times = switch_times
        self.time = 0
    def assign_reward(self, action):
        if self.time in self.switch_times:
            self.prob_action_0 = 1 - self.prob_action_0
        self.time += 1
        return FixedSlotMachineEnvironment.assign_reward(self, action)
