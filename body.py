class Body:
    def __init__(self, **parameters):
        raise NotImplementedError
    def perceive_reward(self, reward):
        ''' maps a reward to a perceived reward '''
        raise NotImplementedError
    def report_fitness(self):
        ''' returns the accumulated fitness over the lifespan '''
        raise NotImplementedError

class NullBody(Body):
    def __init__(self):
        self.total_reward = 0
        self.time = 0
    def perceive_reward(self, reward):
        self.total_reward += reward
        self.time += 1
        return reward
    def report_fitness(self):
        return self.total_reward / self.time

class SatiationBody(NullBody):
    def __init__(self, capacity=10, optimum=8, digestion=.6):
        NullBody.__init__(self)

        self.stomach_max = capacity
        self.optimum = optimum
        self.digestion = digestion

        self.stomach = optimum

    def digest(self, reward):
        self.stomach += reward
        self.stomach -= self.digestion

        if self.stomach < 0:
            self.stomach = 0
        elif self.stomach > self.stomach_max:
            self.stomach = self.stomach_max
    def perceive_reward(self, reward):
        self.digest(reward)
        return NullBody.perceive_reward(self, reward)

class SatiationBody_Reward(SatiationBody):
    def perceive_reward(self, reward):
        self.digest(reward)

        if self.stomach <= self.optimum:
            return NullBody.perceive_reward(self, reward)
        else:
            return NullBody.perceive_reward(self, 0)

class SatiationBody_Fitness(SatiationBody):
    def report_fitness(self):
        return -(self.stomach - self.optimum)**2

class SatiationBody_Both(SatiationBody_Fitness, SatiationBody_Reward):
    pass
