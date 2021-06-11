import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Highest Reward Actions
def max_learning_actions(acts, dur, crit_mat, confidence_interval=None):
    learning = crit_mat[:, 0].flatten()    
    max_reward = max(learning)
    index = 0
    for i in range(len(learning)):
        if max_reward == learning[i]:
            index = i
    actions(acts, dur, index, confidence_interval)
    plt.title("Max Learning: Actions Over Time")
    plt.show()
    
    
# Plot Highest Burst Actions
def max_burst_actions(acts, dur, crit_mat, confidence_interval=None):
    bursts = crit_mat[:, 1].flatten()
    max_burst = max(bursts)
    index = 0
    for i in range(len(bursts)):
        if max_burst == bursts[i]:
            index = i
    actions(acts, dur, index, confidence_interval)
    plt.title("Max Burst: Actions Over Time")
    plt.show()
            
# Plot Highest Decay Actions
def max_decay_actions(acts, dur, crit_mat, confidence_interval=None):
    decay = crit_mat[:, 2].flatten()
    max_decay = max(decay)
    index = 0
    for i in range(len(decay)):
        if max_decay == decay[i]:
            index = i
    actions(acts, dur, index, confidence_interval)
    plt.title("Max Decay: Actions Over Time")
    plt.show()
    
    
# Plot Actions Over Time
def actions(acts, dur, index, confidence_interval=None):
    x = np.linspace(0, dur-1, dur)
    if confidence_interval is None:
        plt.plot(x, acts[index])
    else:
        fig, ax = plt.subplots()
        ax.plot(x, acts[index])
        lower = acts[index] - confidence_interval[index]
        upper = acts[index] + confidence_interval[index]
        ax.fill_between(x, lower, upper, color="b", alpha=.1)
    plt.xlabel("Time")
    plt.ylabel("Actions")
    plt.title("Actions ")
    

# Plot Color Map
def color_map(crit_mat, resolution, continuous=True, growth_minmax=[0, .4], decay_minmax=[.25, 1]):
    if continuous:
        crit_mat = crit_mat.reshape((resolution, resolution, 3))
        plt.imshow(crit_mat)
        plt.title("Continuous Color Map")
    else:
        plot_mat = np.zeros((resolution*resolution))
        for i in range((resolution*resolution)):
            if crit_mat[i, 0] == 0 and crit_mat[i, 1] == 0 and  crit_mat[i, 2] == 0:
                plot_mat[i] = 0
            if crit_mat[i, 0] == 1 and crit_mat[i, 1] == 0 and  crit_mat[i, 2] == 0:
                plot_mat[i] = 1
            if crit_mat[i, 0] == 0 and crit_mat[i, 1] == 1 and  crit_mat[i, 2] == 0:
                plot_mat[i] = 2
            if crit_mat[i, 0] == 0 and crit_mat[i, 1] == 0 and  crit_mat[i, 2] == 1:
                plot_mat[i] = 3
            if crit_mat[i, 0] == 1 and crit_mat[i, 1] == 1 and  crit_mat[i, 2] == 0:
                plot_mat[i] = 4
            if crit_mat[i, 0] == 1 and crit_mat[i, 1] == 0 and  crit_mat[i, 2] == 1:
                plot_mat[i] = 5
            if crit_mat[i, 0] == 0 and crit_mat[i, 1] == 1 and  crit_mat[i, 2] == 1:
                plot_mat[i] = 6
            if crit_mat[i, 0] == 1 and crit_mat[i, 1] == 1 and crit_mat[i, 2] == 1:
                plot_mat[i] = 7
        plot_mat = plot_mat.reshape((resolution, resolution))
        sns.heatmap(plot_mat).set_title("Binary Criteria Color Map")
    plt.xlabel("Expectation Growth")
    plt.ylabel("Expectation Decay")
    plt.show()
        
    
# Plot Param Sweep AUC
def auc(AUC):
    sns.heatmap(AUC).set_title("Burst Size Heat Map")
    plt.xlabel("Expectation Growth")
    plt.ylabel("Expectation Decay")
    plt.show()

# Plot Param Sweep Reward
def reward(reward):
    sns.heatmap(reward).set_title("Fitness Heat Map")
    plt.xlabel("Expectation Growth")
    plt.ylabel("Expectation Decay")
    plt.show()

