import matplotlib.pyplot as plt
import numpy as np
import agents
import bodies
import environments

class Simulation:
    def __init__(self, agent, body, environment):
        self.agent = agent
        self.body = body
        self.environment = environment

    def step(self, track_actions=True):
        action = self.agent.select_action()
        reward = self.environment.assign_reward(action)
        perceived_reward = self.body.perceive_reward(reward)
        self.agent.update(action, perceived_reward)

        if track_actions:
            self.actions[self.time] = action
            self.time += 1

    def run(self, duration, track_actions=True):
        expectations_one = []
        expectations_two = []
        if track_actions:
            self.actions = np.zeros(duration, dtype=np.int32)
            self.time = 0

        for timestep in range(duration):
            self.step(track_actions)
            expectations_one.append(self.agent.expectation[0])
            expectations_two.append(self.agent.expectation[1])
            
        fitness = self.body.report_fitness()

        if track_actions:
            return fitness, self.actions, None, expectations_one, expectations_two

##########################################################
### Code to simulate a single trial, calculate         ###
### fitness and extinction burst size, and plot these. ###
##########################################################    

def simulate(agent_initializer, body_initializer, environment_initializer, dur=600, extinction_begin=200, extinction_end=500, n=500, plot=False):
    total_actions = np.zeros(dur, dtype=np.float32)
    total_fitness = 0
    total_expectations_one = np.zeros(dur)
    total_expectations_two = np.zeros(dur)
    
    for i in range(n):
        agent = eval(agent_initializer)
        body = eval(body_initializer)
        environment = eval(environment_initializer)
        simulation = Simulation(agent, body, environment)
        fitness, actions, rewards, e_o, e_t = simulation.run(dur, track_actions=True)
        total_actions += actions
        total_fitness += fitness
        total_expectations_one += e_o
        total_expectations_two += e_t
    
    total_fitness /= n
    total_expectations_one /= n
    total_expectations_two /= n
    # get proportion of time action is 0
    total_actions = n - total_actions
    # calculate confidence interval using Wald Technique
    adj_prop = np.zeros(dur)
    for i in range(len(adj_prop)):
        if total_actions[i] / n < .5:
            ta = n - total_actions[i]
        else:
            ta = total_actions[i]
        adj_prop[i] = (ta + 2) / (n + 4)
    total_actions /= n
    adj_prop *= (1 - adj_prop)
    div = adj_prop / (n + 4)
    s = np.sqrt(div)
    ci = s * 1.96

    action_at_switch = total_actions[extinction_begin]
    AUC = np.sum(np.clip(total_actions[extinction_begin:extinction_end] - action_at_switch, 0, None))

    if plot:
        plt.axvline(extinction_begin, ls='--', color='lightgray')
        if extinction_end != dur:
            plt.axvline(extinction_end, ls='--', color='lightgray')
        extinction_phase = ((np.arange(dur) > extinction_begin) & (np.arange(dur) < extinction_end) & (total_actions > action_at_switch))
        plt.fill_between(np.arange(dur), total_actions, action_at_switch, where=extinction_phase, facecolor='lightgray', interpolate=True)
        plt.plot(total_actions)
        plt.title(f'Reward: {total_fitness:.2f}, burst: {AUC:.2f}')
        plt.xlabel("Timestep")
        plt.ylabel("Proportion favored arm")

    return total_fitness, AUC, ci, total_actions, total_expectations_one, total_expectations_two
    
            