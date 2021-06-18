#ifndef BODIES_H
#define BODIES_H

#include "include/pcg_random.hpp"

class Body
{
public:
    virtual ~Body();
    virtual void reset(int seed) = 0;
    virtual double perceive_reward(double reward) = 0;
    virtual double get_fitness(int length) = 0;
};

class NullBody : public Body
{
    double m_total_reward;

public:
    ~NullBody();

    void reset(int seed);

    double perceive_reward(double reward);

    double get_fitness(int length);
};

class InfoGainBody : public Body
{
    double m_total_reward;
    const int m_extinction_begin;
    int m_time;

public:
    InfoGainBody(int extinction_begin);

    ~InfoGainBody();
    
    void reset(int seed);

    double perceive_reward(double reward);

    double get_fitness(int length);
};

class NoisyBody : public Body
{
    double m_total_reward;
    const uint32_t m_switch_prob;
    bool m_reward_access = true;

    pcg32 m_rand;

public:
    NoisyBody(double switch_prob);

    ~NoisyBody();

    void reset(int seed);

    double perceive_reward(double reward);

    double get_fitness(int length);
};

#endif // #ifndef BODIES_H
