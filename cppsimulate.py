import numpy as np

from extinctionbursts import *

def cpp_simulate(agent_initializer, body_initializer, environment_initializer, dur=600, extinction_begin=200, extinction_end=500, n=500, plot=False):
    agent = eval(agent_initializer)
    body = eval(body_initializer)
    environment = eval(environment_initializer)

    total_fitness, action_probs = simulate(n, dur, 42, agent, body, environment)

    total_actions = action_probs * n

    # calculate confidence interval using Wald Technique
    adj_prop = np.zeros(dur)
    for i in range(len(adj_prop)):
        if total_actions[i] / n < .5:
            ta = n - total_actions[i]
        else:
            ta = total_actions[i]
        adj_prop[i] = (ta + 2) / (n + 4)
    adj_prop *= (1 - adj_prop)
    div = adj_prop / (n + 4)
    s = np.sqrt(div)
    ci = s * 1.96

    action_at_switch = action_probs[extinction_begin]
    AUC = np.sum(np.clip(action_probs[extinction_begin:extinction_end] - action_at_switch, 0, None))

    if plot:
        import matplotlib.pyplot as plt

        plt.axvline(extinction_begin, ls='--', color='lightgray')
        if extinction_end != dur:
            plt.axvline(extinction_end, ls='--', color='lightgray')
        extinction_phase = ((np.arange(dur) > extinction_begin) & (np.arange(dur) < extinction_end) & (action_probs > action_at_switch))
        plt.fill_between(np.arange(dur), action_probs, action_at_switch, where=extinction_phase, facecolor='lightgray', interpolate=True)
        plt.plot(action_probs)
        plt.title(f'Reward: {total_fitness:.2f}, burst: {AUC:.2f}')
        plt.xlabel("Timestep")
        plt.ylabel("Proportion favored arm")

    return total_fitness, AUC, ci, action_probs

if __name__ == '__main__':
    f, AUC, ci, action_probs = cpp_simulate('FrustrationModelAgent(.3, .9, .1, .1)', 'NullBody()', 'TrueExtinctionEnvironment(.7, 300, 400)', 500, 300, 400, 500, True)
    import matplotlib.pyplot as plt
    plt.show()
