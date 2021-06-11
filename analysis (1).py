import simulate
import agents
import environments
import bodies
import plot
import cppsimulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import time
    
def four_D(resolution, body_initializer, environment_initializer, reward_vals = [.3], cost_vals = [1], growth_minmax=[0, 1], decay_minmax=[0, 1], switch_times=[200, 500], cont=True, n=500):
    fitnesses = []
    AUCs = []
    actions = []
    CIs = []
    critmats = []
    
    decay_vals = np.linspace(decay_minmax[0], decay_minmax[1], resolution)
    growth_vals = np.linspace(growth_minmax[0], growth_minmax[1], resolution)
    
    exp_decay = np.array([])
    exp_growth = np.array([])
    reward = np.array([])
    cost = np.array([])
    
    # get all combinations of reward and cost
    for i in range(len(cost_vals)):
        cost = np.append(cost, cost_vals)
    for i in range(len(reward_vals)):
        for j in range(len(reward_vals)):
            reward = np.append(reward, reward_vals[i])
    
    # get all combinations of expectation growth and decay
    for i in range(resolution):
        exp_decay = np.append(exp_decay, decay_vals)
    for i in range(resolution):
        for j in range(resolution):
            exp_growth = np.append(exp_growth, growth_vals[i])
    
    for i in range(len(reward)):
        all_fitness = np.zeros((resolution, resolution))
        all_AUC = np.zeros((resolution, resolution))
        tot_actions = []
        cis = []
        r = reward[i]
        c = cost[i]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(cpp_sweep, zip(exp_growth , exp_decay, multiprocessing.repeat(r), multiprocessing.repeat(c), multiprocessing.repeat(body_initializer), multiprocessing.repeat(environment_initializer), multiprocessing.repeat(n)))
        results = np.array(results)
        results = results.reshape((resolution, resolution))
        for k in range(resolution):
            for j in range(resolution):
                all_fitness[k, j] = results[k, j].get("fitness")
                all_AUC[k, j] = results[k, j].get("AUC")
                tot_actions.append(results[k, j].get("total_actions"))
                cis.append(results[k, j].get("ci"))
        if cont:
            crit_mat = criteria_mat(tot_actions, all_AUC, resolution, cis, switch_times[0], switch_times[1])
        if not cont: 
            crit_mat = criteria_mat(tot_actions, all_AUC, resolution, cis, switch_times[0], switch_times[1], cont_crit=True)
        fitnesses.append(all_fitness)
        AUCs.append(all_AUC)
        actions.append(tot_actions)
        CIs.append(cis)
        critmats.append(crit_mat)
    
    return fitnesses, AUCs, actions, CIs, critmats

def parameter_sweep(reward, cost, resolution, body_initializer, environment_initializer, simulation_kwargs={}, plot=False, switch_times=[200, 500], cont=True, growth_minmax=[0, 1], decay_minmax=[0, 1], n=500):
    exp_decay = np.linspace(decay_minmax[0], decay_minmax[1], resolution)
    exp_growth = np.linspace(growth_minmax[0], growth_minmax[1], resolution)
    all_fitness = np.zeros((resolution, resolution))
    all_AUC = np.zeros((resolution, resolution))
    tot_actions = []
    cis = []
    tot_expectations_one = []
    tot_expectations_two = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(sweep, itertools.product(exp_growth, exp_decay, [reward], [cost], [body_initializer], [environment_initializer], [n]))
    results = np.array(results)
    results = results.reshape((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            all_fitness[i, j] = results[i, j].get("fitness")
            all_AUC[i, j] = results[i, j].get("AUC")
            tot_actions.append(results[i, j].get("total_actions"))
            tot_expectations_one.append(results[i, j].get("total_expectations_one"))
            tot_expectations_two.append(results[i, j].get("total_expectations_two"))
            cis.append(results[i, j].get("ci"))
    crit_mat = criteria_mat(tot_actions, all_AUC, resolution, cis, switch_times[0], switch_times[1], cont=cont)
    if plot:
        # pcolormesh expects the boundaries between quads, not just a single corner of each quad
        plot_exp_growth, plot_exp_decay = np.meshgrid(
            np.linspace(np.min(exp_growth), np.max(exp_growth), resolution+1),
            np.linspace(np.min(exp_decay), np.max(exp_decay), resolution+1)
            )
        fig, ((fitness_plot, AUC_plot, corr_plot), (sample_plot_reward, sample_plot_burst, sample_plot_bad)) = plt.subplots(2, 3)
        # plt.suptitle(f"Parameter sweep with reward {reward}, cost {cost}, {body_initializer}, {environment_initializer}")

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
        print(all_fitness)
        print(all_AUC)
        best_reward_indices = np.unravel_index(np.argmax(all_fitness), all_fitness.shape)
        best_AUC_indices = np.unravel_index(np.argmax(all_AUC), all_AUC.shape)
        worst_reward_indices = np.unravel_index(np.argmax(-all_fitness), all_fitness.shape)
        print(best_reward_indices)
        print(best_AUC_indices)
        print(worst_reward_indices)

        plt.sca(sample_plot_reward)
        this_exp_growth = exp_growth[best_reward_indices[0]]
        this_exp_decay = exp_decay[best_reward_indices[1]]
        agent_initializer = f'''agents.FrustrationModelAgent(reward={reward}, cost={cost},
        expectation_growth={this_exp_growth}, expectation_decay={this_exp_decay})'''
        fitness, AUC, _, _, _, _ = simulate.simulate(agent_initializer, body_initializer, environment_initializer, plot=True, **simulation_kwargs)
        plt.title(f'Best fitness at {this_exp_growth:.2f}, {this_exp_decay:.2f}\nFitness: {fitness:.2f}, burst: {AUC:.2f}')

        plt.sca(sample_plot_burst)
        this_exp_growth = exp_growth[best_AUC_indices[0]]
        this_exp_decay = exp_decay[best_AUC_indices[1]]
        agent_initializer = f'''agents.FrustrationModelAgent(reward={reward}, cost={cost},
        expectation_growth={this_exp_growth}, expectation_decay={this_exp_decay})'''
        fitness, AUC, _, _, _, _ = simulate.simulate(agent_initializer, body_initializer, environment_initializer, plot=True, **simulation_kwargs)
        plt.title(f'Biggest burst at {this_exp_growth:.2f}, {this_exp_decay:.2f}\nFitness: {fitness:.2f}, burst: {AUC:.2f}')

        plt.sca(sample_plot_bad)
        this_exp_growth = exp_growth[worst_reward_indices[0]]
        this_exp_decay = exp_decay[worst_reward_indices[1]]
        agent_initializer = f'''agents.FrustrationModelAgent(reward={reward}, cost={cost},
        expectation_growth={this_exp_growth}, expectation_decay={this_exp_decay})'''
        fitness, AUC, _, _, _, _ = simulate.simulate(agent_initializer, body_initializer, environment_initializer, plot=True, **simulation_kwargs)
        plt.title(f'Worst fitness at {this_exp_growth:.2f}, {this_exp_decay:.2f}\nFitness: {fitness:.2f}, burst: {AUC:.2f}')

        plt.tight_layout()

    return all_fitness, all_AUC, cis, tot_actions, tot_expectations_one, tot_expectations_two, crit_mat

def sweep(g, d, reward, cost, body_init, env_init, n):
    agent_initializer = f'''agents.FrustrationModelAgent(reward={reward}, cost={cost},
                            expectation_growth={g}, expectation_decay={d})'''
    body_initializer = body_init
    environment_initializer = env_init
    fitness, AUC, ci, total_actions, total_expectations_one, total_expectations_two = simulate.simulate(agent_initializer, body_initializer, environment_initializer, n=n)
    
    return {"fitness":fitness, "AUC":AUC, "ci":ci, "total_actions":total_actions, "total_expectations_one":total_expectations_one, "total_expectations_two":total_expectations_two}

def cpp_sweep(g, d, reward, cost, body_init, env_init, n):
    agent_initializer = f'''FrustrationModelAgent({reward}, {cost},
                            {g}, {d})'''
    body_initializer = body_init.replace('bodies.', '')
    environment_initializer = env_init.replace('environments.', '')
    fitness, AUC, ci, total_actions = cppsimulate.cpp_simulate(agent_initializer, body_initializer, environment_initializer, n=n)

    return {"fitness":fitness, "AUC":AUC, "ci":ci, "total_actions":total_actions}

def cont_crit(act_begin, act_end, AUC, max_AUC):
    crit = np.zeros(3)
    if act_begin > .5:
        crit[0] = (act_begin - .5) * 2
    if act_begin < .5:
        crit[0] = 0
    crit[1] = AUC / max_AUC
    if act_begin > act_end:
        crit[2] = (act_begin - act_end) 
    if act_begin < act_end:
        crit[2] = 0 
    return crit

def binary_crit(act_begin, act_end, ci, AUC):
    crit = np.zeros(3)
    if act_begin - ci > .5:
        crit[0] = 1
    if AUC > .05:
        crit[1] = 1
    if act_begin - ci > act_end:
        crit[2] = 1
    return crit


def criteria_mat(actions, auc, res, cis, extinction_begin, extinction_end, cont=True):
    crit_mat = np.zeros((res*res, 3))
    actions = np.array(actions)
    cis = np.array(cis)
    auc = auc.flatten()
    max_AUC = max(auc)
    for i in range(res*res):
        act_begin = actions[i, extinction_begin] 
        act_end = actions[i, extinction_end]
        ci = cis[i, extinction_begin]
        AUC = auc[i]
        if cont:
            crit = cont_crit(act_begin, act_end, AUC, max_AUC)
        else:    
            crit = binary_crit(act_begin, act_end, ci, AUC)
        crit_mat[i] = crit
    return crit_mat.reshape((res*res, 3))
        

if __name__ == '__main__':
    save_data = False    # save output
    two_d = True   # True = 2D Param Sweep | False = 4D Param Sweep
    body = 'bodies.NullBody()'  # Body Initializer
    env = 'environments.TrueExtinctionEnvironment(.6, 200, 500)'  # Environment Intializer
    switch_times = [200, 500] 
    cont = True     # True if criteria map is continuous
    growth_minmax = [0, .4]     # Min and Max for growth values
    decay_minmax = [.25, 1]     # Min and Max for decay values
    res = 100    # Growth and Decay Resolution
    n = 10_000     # Number of Simulations
    plot = True

    sweep = cpp_sweep
    
    ######## 2D PARAMETER SWEEP ########
    reward = .3
    cost = 1

    if two_d:
        tic = time.perf_counter()
        fit, auc, cis, total_actions, tot_expectations_one, tot_expectations_two, crit_mat = parameter_sweep(reward, cost, res, body, env, {'extinction_end': switch_times[1]}, plot=False, switch_times=switch_times, cont=cont, growth_minmax=growth_minmax, decay_minmax=decay_minmax, n=n)
        toc = time.perf_counter()
        print("2-D Time: " + str(toc - tic))
        
        if plot:
            plot.max_learning_actions(total_actions, 600, crit_mat, confidence_interval=cis)
            plot.max_burst_actions(total_actions, 600, crit_mat, confidence_interval=cis)
            plot.max_decay_actions(total_actions, 600, crit_mat, confidence_interval=cis)
            plot.color_map(crit_mat, res, continuous=cont)
            plot.auc(auc)
            plot.reward(fit)
    
    ######## 4-DIMENSIONAL ANALYSIS ########
    '''
    The resolution for reward and cost will be decided by the length of the cost and reward values lists.
    The length of these lists must be equal.
    If the length of reward and cost values is 3, then 9 (3**2) pairings will be tested
    '''
    # the Reward and Cost values you want to test
    reward_vals = [.1, 1.] 
    cost_vals = [.1, 1.] 

    if not two_d:
        tic = time.perf_counter()
        fitnesses, AUCs, actions, CIs, critmats = four_D(res, body, env, reward_vals, cost_vals, growth_minmax, decay_minmax, switch_times, cont=cont, n=n)
        toc = time.perf_counter()
        print("4-D Time: " + str(toc - tic))
    ### SAVE DATA ###
    ####### PARAMETER SWEEP #######
    if save_data:
        if two_d: 
            fit = pd.DataFrame(fit.flatten())
            auc = pd.DataFrame(auc.flatten())
    
            fit.to_csv("baseline_results\baseline_fit_6.csv")
            auc.to_csv("baseline_results\baseline_AUC_6.csv")
            
            tot_actions = pd.DataFrame(np.array(total_actions))
            tot_actions.to_csv("baseline_results\baseline_actions_6.csv")

            cis = pd.DataFrame(np.array(cis))
            cis.to_csv("baseline_results\baseline_cis_6.csv")
            
            critmat = pd.DataFrame(crit_mat)
            critmat.to_csv("baseline_results\baseline_critmat_6.csv")            

        if not two_d:
            cost = np.array([])
            reward = np.array([])
            
            for i in range(len(cost_vals)):
                cost = np.append(cost, cost_vals)
            for i in range(len(reward_vals)):
                for j in range(len(reward_vals)):
                    reward = np.append(reward, reward_vals[i])
                    
            for i in range(len(reward)):
                curr_fitness = fitnesses[i]
                curr_AUC = AUCs[i]
                curr_actions = actions[i]
                curr_ci = CIs[i]
                curr_critmat = critmats[i]
                
                curr_fitness = pd.DataFrame(curr_fitness.flatten())
                curr_AUC = pd.DataFrame(curr_AUC.flatten())
                curr_actions = pd.DataFrame(curr_actions)
                curr_ci = pd.DataFrame(curr_ci)
                curr_critmat = pd.DataFrame(curr_critmat)
                
                fit_fn = "fit_reward_{}_cost_{}.csv".format(reward[i], cost[i])
                auc_fn = "auc_reward_{}_cost_{}.csv".format(reward[i], cost[i])
                act_fn = "act_reward_{}_cost_{}.csv".format(reward[i], cost[i])
                ci_fn = "ci_reward_{}_cost_{}.csv".format(reward[i], cost[i])
                critmat_fn = "critmat_reward_{}_cost_{}.csv".format(reward[i], cost[i])
                
                curr_fitness.to_csv(fit_fn)
                curr_AUC.to_csv(auc_fn)
                curr_actions.to_csv(act_fn)
                curr_ci.to_csv(ci_fn)
                curr_critmat.to_csv(critmat_fn)
                    
                    
