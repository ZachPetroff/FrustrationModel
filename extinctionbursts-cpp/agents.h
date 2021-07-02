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

class UncertaintyModelAgent : public Agent
{
    const double m_w_uncertainty;
    const double m_fast_rate;
    const double m_slow_rate;

    const double m_exponent;

    pcg32 m_rand;

public:
    double m_fast_expectation[2];
    double m_slow_expectation[2];

    UncertaintyModelAgent(double w_uncertainty,
        double fast_lambda, double slow_lambda,
        double temperature=1.);

    ~UncertaintyModelAgent();

    void reset(int seed);

    inline double rate(int i)
    {
        const double err = m_fast_expectation[i] - m_slow_expectation[i];
        return m_fast_expectation[i] + m_w_uncertainty*err*err;

        // err = (old - delta) - old = -delta;
        // (old - delta) + uncertainty_weight*err*err;
        // old - delta + w_uncertainty*delta**2
        // w_uncertainty*delta**2 > delta?

        // produce plots explaining when this produces a burst
    }

    int select_action();

    void update(int action, double reward_signal);
};

class UncertaintyModelAgent2 : public Agent
{
    const double m_w_uncertainty;
    const double m_reward_rate;
    const double m_change_rate;

    const double m_exponent;

    pcg32 m_rand;

public:
    double m_reward_expectation[2];
    double m_change_expectation[2];

    UncertaintyModelAgent2(double w_uncertainty,
        double reward_lambda, double change_lambda,
        double temperature=1.);

    ~UncertaintyModelAgent2();

    void reset(int seed);

    inline double rate(int i)
    {
        if(m_change_expectation[i] > 0)
            return m_reward_expectation[i] + m_w_uncertainty*m_change_expectation[i];
        else
            return m_reward_expectation[i] - m_w_uncertainty*m_change_expectation[i];
    }

    int select_action();

    void update(int action, double reward_signal);
};

#endif // #ifndef AGENTS_H
