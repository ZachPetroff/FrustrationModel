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

#define LN2 0.6931471805599453

// UncertaintyModelAgent implementation
UncertaintyModelAgent::UncertaintyModelAgent(double w_uncertainty,
        double fast_lambda, double slow_lambda, double temperature) :
    m_w_uncertainty(w_uncertainty),
    m_fast_rate(LN2/fast_lambda),
    m_slow_rate(LN2/slow_lambda),
    m_exponent(1/temperature) {}

UncertaintyModelAgent::~UncertaintyModelAgent() {}

void UncertaintyModelAgent::reset(int seed)
{
    m_rand = pcg32(seed);
    for(int i = 0; i < 2; i++)
    {
        m_fast_expectation[i] = 0.5;
        m_slow_expectation[i] = 0.5;
    }
}

int UncertaintyModelAgent::select_action()
{
    double e_0 = pow(rate(0), m_exponent);
    double den = e_0 + pow(rate(1), m_exponent);
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

void UncertaintyModelAgent::update(int action, double perceived_reward)
{
    m_fast_expectation[action] += (perceived_reward-m_fast_expectation[action])*m_fast_rate;
    m_slow_expectation[action] += (perceived_reward-m_slow_expectation[action])*m_slow_rate;
}

// UncertaintyModelAgent2 implementation
UncertaintyModelAgent2::UncertaintyModelAgent2(double w_uncertainty,
        double reward_lambda, double change_lambda, double temperature) :
    m_w_uncertainty(w_uncertainty),
    m_reward_rate(LN2/reward_lambda),
    m_change_rate(LN2/change_lambda),
    m_exponent(1/temperature) {}

UncertaintyModelAgent2::~UncertaintyModelAgent2() {}

void UncertaintyModelAgent2::reset(int seed)
{
    m_rand = pcg32(seed);
    for(int i = 0; i < 2; i++)
    {
        m_reward_expectation[i] = 0.5;
        m_change_expectation[i] = 0.;
    }
}

int UncertaintyModelAgent2::select_action()
{
    double e_0 = pow(rate(0), m_exponent);
    double den = e_0 + pow(rate(1), m_exponent);
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

void UncertaintyModelAgent2::update(int action, double perceived_reward)
{
    double delta = (perceived_reward-m_reward_expectation[action])*m_reward_rate;
    m_reward_expectation[action] += delta;
    m_change_expectation[action] += (delta-m_change_expectation[action])*m_change_rate;
}
