#ifndef SIMULATE_H
#define SIMULATE_H

#include "agents.h"
#include "bodies.h"
#include "environments.h"

struct SimulationResult
{
    double* action_freq;
    double fitness;
};

void simulate(int n, int duration, int seed, Agent& agent, Body& body,
    Environment& environment, SimulationResult& result);

#endif // #ifndef SIMULATE_H
