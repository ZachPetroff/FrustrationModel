#ifndef SIMULATE_H
#define SIMULATE_H

#include "agents.h"
#include "bodies.h"
#include "environments.h"

struct SimulationResult
{
    double* action_freq;
    double* slow_expectation_0;
    double* fast_expectation_0;
    double* slow_expectation_1;
    double* fast_expectation_1;
    double fitness;
};

void simulate(int n, int duration, int seed, Agent& agent, Body& body,
    Environment& environment, SimulationResult& result);

#endif // #ifndef SIMULATE_H
