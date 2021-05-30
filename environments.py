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
        
class TrueExtinctionEnvironment(Environment):
    def __init__(self, prob_action_0, switch_times):
        self.prob_action_0 = prob_action_0
        self.switch_times = switch_times
        self.time = 0
    def assign_reward(self, action):
        if self.time > self.switch_times[0] and self.time < self.switch_times[1]:
            returnVal = 0
        else:
            returnVal = FixedSlotMachineEnvironment.assign_reward(self, action)
        self.time += 1
        return returnVal

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
    
class PerArmSlotMachineEnvironment(FixedSlotMachineEnvironment):
    def __init__(self, prob_action_0, switch_likelihood, switch_times):
        self.prob_action_0 = prob_action_0
        self.switch_likelihood = switch_likelihood
        self.time = 0
        self.switch_times = switch_times
    def assign_reward(self, action):
        if self.switch_times[0] < self.time and self.time < self.switch_times[1]:
            switch = random.random()
            if switch < self.switch_likelihood:
                self.prob_action_0 = 1 - self.prob_action_0
        self.time += 1
        return FixedSlotMachineEnvironment.assign_reward(self, action)
            
class OneArmedEnvironment(Environment):
    def __init__(self, prob_action, switch_times, combination):
        self.prob_action = prob_action
        self.switch_times = switch_times
        self.combination = combination
        self.time = 0
        
    def assign_reward(self, action):
        reward_action = random.random() < self.prob_action
        if self.time in self.switch_times:
            self.prob_action = 1 - self.prob_action
        self.time += 1
        # Moves and reward is given
        if not action and reward_action:
            return self.combination[0]
        # Moves and reward is not given
        if not action and not reward_action:
            return self.combination[1]
        # Does not move and reward is present
        if action and reward_action:
            return self.combination[2]
        # Does not move and reward is not present
        if action and not reward_action:
            return self.combination[3]

class TimeDelayEnvironment(OneArmedEnvironment):
    def __init__(self, delay, switch_times):
        self.delay = delay
        self.switch_times = switch_times
        self.history = []
        self.time = 0
        self.prob_action = 1
        self.combination = [1, -1, 0, 0]
        
    def assign_reward(self, action):
        if self.time in self.switch_times:
            self.prob_action = 1 - self.prob_action
        if sum(self.history[-(self.delay+1):-1]) >= 1:
            reward = 0
        else:
            reward = OneArmedEnvironment.assign_reward(self, action)
        self.history.append(reward)
        self.time += 1
        return reward