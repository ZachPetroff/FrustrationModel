#include "simulate.h"

static void simulate_single_run(int duration, int seed,
    Agent& agent, Body& body, Environment& environment,
    double* total_actions, double* total_fast_exp_0, double* total_slow_exp_0,
    double* total_fast_exp_1, double* total_slow_exp_1, double& total_fitness)
{
    agent.reset(seed);
    body.reset(seed^0x1d4abf8c);
    environment.reset(seed^0xf2b8eca0);
    for(int i = 0; i < duration; i++)
    {
        int action = agent.select_action();
        // this is our convention
        if(action == 0)
            total_actions[i] += 1;

        UncertaintyModelAgent* agent_u = dynamic_cast<UncertaintyModelAgent*>(&agent);
        if(agent_u)
        {
            total_fast_exp_0[i] += agent_u->m_fast_expectation[0];
            total_fast_exp_1[i] += agent_u->m_fast_expectation[1];
            total_slow_exp_0[i] += agent_u->m_slow_expectation[0];
            total_slow_exp_1[i] += agent_u->m_slow_expectation[1];
        }

        double reward = environment.assign_reward(action);
        double perceived_reward = body.perceive_reward(reward);
        agent.update(action, perceived_reward);
    }
    total_fitness += body.get_fitness(duration);
}

void simulate(int n, int duration, int seed, Agent& agent, Body& body,
    Environment& environment, SimulationResult& result)
{
    result.fitness = 0;
    for(int i = 0; i < duration; i++)
    {
        result.action_freq[i] = 0;
        result.slow_expectation_0[i] = 0;
        result.slow_expectation_1[i] = 0;
        result.fast_expectation_0[i] = 0;
        result.fast_expectation_1[i] = 0;
    }

    for(int i = 0; i < n; i++)
    {
        simulate_single_run(duration, seed+i*123, agent, body, environment,
            result.action_freq, result.fast_expectation_0, result.slow_expectation_0,
            result.fast_expectation_1, result.slow_expectation_1, result.fitness);
    }

    for(int i = 0; i < duration; i++)
    {
        result.action_freq[i] /= n;
        result.slow_expectation_0[i] /= n;
        result.slow_expectation_1[i] /= n;
        result.fast_expectation_0[i] /= n;
        result.fast_expectation_1[i] /= n;
    }

    result.fitness /= n;
}
