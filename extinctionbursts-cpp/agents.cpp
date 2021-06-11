#include <cmath>

#include "agents.h"

Agent::~Agent() {}

// FrustrationModelAgent implementation
FrustrationModelAgent::FrustrationModelAgent(double reward, double cost,
    double expectation_growth, double expectation_decay,
    double temperature) :
    m_reward(reward), m_cost(cost),
    m_expectation_growth(expectation_growth),
    m_expectation_decay(expectation_decay),
    m_exponent(1/temperature) {}

FrustrationModelAgent::~FrustrationModelAgent() {}

void FrustrationModelAgent::reset(int seed)
{
    m_rand = pcg32(seed);
    for(int i = 0; i < 2; i++)
    {
        m_policy[i] = 0.;
        m_expectation[i] = 0.;
    }
}

int FrustrationModelAgent::select_action()
{
    double e_0 = pow(m_policy[0], m_exponent);
    double den = e_0 + pow(m_policy[1], m_exponent);
    double p_0;
    if(den == 0)
        p_0 = 0.5;
    else
        p_0 = e_0 / den;
    double r = ((double)m_rand() / (double)0x100000000);
    if(r < p_0)
        return 0;
    else
        return 1;
}

void FrustrationModelAgent::update(int action, double perceived_reward)
{
    if(perceived_reward > 0)
    {
        m_policy[action] += m_reward * perceived_reward;
        m_policy[action] -= m_expectation[action];
        if(m_policy[action] < 0.)
            m_policy[action] = 0.;
        m_expectation[action] += m_expectation_growth;
    }
    else
    {
        m_policy[action] -= m_cost;
        m_policy[action] += m_expectation[action];
        if(m_policy[action] < 0.)
            m_policy[action] = 0.;
        m_expectation[action] -= m_expectation_decay;
        if(m_expectation[action] < 0.)
            m_expectation[action] = 0.;
    }
}
