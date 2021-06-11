#include "agents.h"
#include "bodies.h"
#include "environments.h"
#include "simulate.h"

int main(int argc, char** argv)
{
    int seed = 42;
    if(argc > 1)
        seed = atoi(argv[1]);
    NullBody body;
    TrueExtinctionEnvironment environment(.7, 300, 400);
    SimulationResult result;
    result.action_freq = new double[500];
    for(double reward = 0.; reward <= 1.; reward += .1)
    {
        for(double cost = 0.; cost <= 1.; cost += .1)
        {
            for(double expectation_growth = 0.; expectation_growth <= 1.; expectation_growth += .1)
            {
                for(double expectation_decay = 0.; expectation_decay <= 1.; expectation_decay += .1)
                {
                    FrustrationModelAgent agent(reward, cost, expectation_growth, expectation_decay);
                    simulate(500, 500, seed, agent, body, environment, result);
                    printf("%f %f %f %f Fitness: %f\n", reward, cost, expectation_growth, expectation_decay, result.fitness);
                }
            }
        }
    }
    // printf("Fitness: %f\n", result.fitness);
    // for(int i = 0; i < 500; i++)
    //     printf("Action at time %d: %f\n", i, result.action_freq[i]);
    delete result.action_freq;
    return 0;
}
