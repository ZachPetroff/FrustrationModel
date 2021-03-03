import matplotlib.pyplot as plt
import numpy as np

class Simulation:
    def __init__(self, agent, body, environment):
        self.agent = agent
        self.body = body
        self.environment = environment

    def step(self, track_actions=False):
        action = self.agent.select_action()
        reward = self.environment.assign_reward(action)
        perceived_reward = self.body.perceive_reward(reward)
        self.agent.update(action, perceived_reward)

        if track_actions:
            self.actions[self.time] = action
            self.time += 1

    def run(self, duration, track_actions=False):
        if track_actions:
            self.actions = np.zeros(duration, dtype=np.int32)
            self.time = 0

        for timestep in range(duration):
            self.step(track_actions)

        fitness = self.body.report_fitness()

        if track_actions:
            return fitness, self.actions
        else:
            return fitness

##########################################################
### Code to simulate a single trial, calculate         ###
### fitness and extinction burst size, and plot these. ###
##########################################################

def simulate(agent_initializer, body_initializer, environment_initializer, dur=200, extinction_begin=100, extinction_end=200, n=100, plot=False):
    total_actions = np.zeros(dur, dtype=np.float32)
    total_fitness = 0
    for i in range(n):
        agent = eval(agent_initializer)
        body = eval(body_initializer)
        environment = eval(environment_initializer)
        body = eval(body_initializer)
        environment = eval(environment_initializer)
        simulation = Simulation(agent, body, environment)
        fitness, actions = simulation.run(dur, track_actions=True)
        total_actions += actions
        total_fitness += fitness
    total_fitness /= n
    # get proportion of time action is 0
    total_actions /= -n
    total_actions += 1

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

    return total_fitness, AUC

###########################################################################
### Code to perform a full parameter sweep and calculate/plot heatmaps. ###
###########################################################################

def parameter_sweep(reward, cost, resolution, body_initializer, environment_initializer, simulation_kwargs={}, plot=False, use_tqdm=True):
    exp_growth = np.linspace(0, 1, resolution)
    exp_decay = np.linspace(0, 1, resolution)
    all_fitness = np.zeros((resolution, resolution))
    all_AUC = np.zeros((resolution, resolution))

    enum_growth = enumerate(exp_growth)
    if use_tqdm:
        from tqdm import tqdm
        enum_growth = tqdm(list(enum_growth))

    for i, g in enum_growth:
        for j, d in enumerate(exp_decay):
            agent_initializer = f'''FrustrationModelAgent(reward={reward}, cost={cost},
                expectation_growth={g}, expectation_decay={d})'''
            fitness, AUC = simulate(agent_initializer, body_initializer, environment_initializer, **simulation_kwargs)
            all_fitness[i,j] = fitness
            all_AUC[i,j] = AUC

    if plot:
        # pcolormesh expects the boundaries between quads, not just a single corner of each quad
        plot_exp_growth, plot_exp_decay = np.meshgrid(
            np.linspace(np.min(exp_growth), np.max(exp_growth), resolution+1),
            np.linspace(np.min(exp_decay), np.max(exp_decay), resolution+1)
        )

        fig, ((fitness_plot, AUC_plot, corr_plot), (sample_plot_reward, sample_plot_burst, sample_plot_bad)) = plt.subplots(2, 3)

        plt.suptitle(f"Parameter sweep with reward {reward}, cost {cost}, {body_initializer}, {environment_initializer}")

        plt.sca(fitness_plot)
        plt.pcolormesh(plot_exp_growth, plot_exp_decay, all_fitness)
        cb = plt.colorbar()
        plt.title("Reward Heatmap")
        plt.xlabel("Expectation Decay")
        plt.ylabel("Expectation Growth")
        cb.set_label("Reward")

        plt.sca(AUC_plot)
        print(plot_exp_growth.shape, all_AUC.shape)
        plt.pcolormesh(plot_exp_growth, plot_exp_decay, all_AUC)
        cb = plt.colorbar()
        plt.title("Extinction Burst Heatmap")
        plt.xlabel("Expectation Decay")
        plt.ylabel("Expectation Growth")
        cb.set_label("Size of Extinction Burst")

        plt.sca(corr_plot)
        plt.plot(all_fitness.flatten(), all_AUC.flatten(), "ro")
        plt.xlabel("Mean Reward")
        plt.ylabel("Size of Burst")
        plt.title("Reward V. Burst")

        # Thank you Sven Marnach! [https://stackoverflow.com/users/279627/sven-marnach]
        # Ref: https://stackoverflow.com/questions/9482550/argmax-of-numpy-array-returning-non-flat-indices
        best_reward_indices = np.unravel_index(np.argmax(all_fitness), all_fitness.shape)
        best_AUC_indices = np.unravel_index(np.argmax(all_AUC), all_AUC.shape)
        worst_reward_indices = np.unravel_index(np.argmax(-all_fitness), all_fitness.shape)

        plt.sca(sample_plot_reward)
        this_exp_growth = exp_growth[best_reward_indices[0]]
        this_exp_decay = exp_decay[best_reward_indices[1]]
        agent_initializer = f'''FrustrationModelAgent(reward={reward}, cost={cost},
            expectation_growth={this_exp_growth}, expectation_decay={this_exp_growth})'''
        fitness, AUC = simulate(agent_initializer, body_initializer, environment_initializer, plot=True, **simulation_kwargs)
        plt.title(f'Best fitness at {this_exp_growth:.2f}, {this_exp_decay:.2f}\nFitness: {fitness:.2f}, burst: {AUC:.2f}')

        plt.sca(sample_plot_burst)
        this_exp_growth = exp_growth[best_AUC_indices[0]]
        this_exp_decay = exp_decay[best_AUC_indices[1]]
        agent_initializer = f'''FrustrationModelAgent(reward={reward}, cost={cost},
            expectation_growth={this_exp_growth}, expectation_decay={this_exp_growth})'''
        fitness, AUC = simulate(agent_initializer, body_initializer, environment_initializer, plot=True, **simulation_kwargs)
        plt.title(f'Biggest burst at {this_exp_growth:.2f}, {this_exp_decay:.2f}\nFitness: {fitness:.2f}, burst: {AUC:.2f}')

        plt.sca(sample_plot_bad)
        this_exp_growth = exp_growth[worst_reward_indices[0]]
        this_exp_decay = exp_decay[worst_reward_indices[1]]
        agent_initializer = f'''FrustrationModelAgent(reward={reward}, cost={cost},
            expectation_growth={this_exp_growth}, expectation_decay={this_exp_growth})'''
        fitness, AUC = simulate(agent_initializer, body_initializer, environment_initializer, plot=True, **simulation_kwargs)
        plt.title(f'Worst fitness at {this_exp_growth:.2f}, {this_exp_decay:.2f}\nFitness: {fitness:.2f}, burst: {AUC:.2f}')

        plt.tight_layout()

    return all_fitness, all_AUC
