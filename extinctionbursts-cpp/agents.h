#ifndef AGENTS_H
#define AGENTS_H

#include "include/pcg_random.hpp"

class Agent
{
public:
    virtual ~Agent();
    virtual void reset(int seed) = 0;
    virtual int select_action() = 0;
    virtual void update(int action, double reward_signal) = 0;
};

class FrustrationModelAgent : public Agent
{
    double m_policy[2];
    double m_expectation[2];

    const double m_reward;
    const double m_cost;
    const double m_expectation_growth;
    const double m_expectation_decay;
    const double m_exponent;

    pcg32 m_rand;

public:
    FrustrationModelAgent(double reward, double cost,
        double expectation_growth, double expectation_decay,
        double temperature=1.);

    ~FrustrationModelAgent();

    void reset(int seed);

    int select_action();

    void update(int action, double reward_signal);
};

#endif // #ifndef AGENTS_H
