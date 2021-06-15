#ifndef ENVIRONMENTS_H
#define ENVIRONMENTS_H

#include "include/pcg_random.hpp"

class Environment
{
public:
    virtual ~Environment();
    virtual void reset(int seed) = 0;
    virtual double assign_reward(int action) = 0;
};

class TrueExtinctionEnvironment : public Environment
{
    const bool m_always_action_0;
    const uint32_t m_action_0_threshold;
    int m_time;
    const int m_extinction_begin;
    const int m_extinction_end;

    pcg32 m_rand;

public:
    TrueExtinctionEnvironment(double p, int extinction_begin,
        int extinction_end);

    ~TrueExtinctionEnvironment();

    void reset(int seed);

    double assign_reward(int action);
};

class SwitchingSlotMachineEnvironment : public Environment
{
    double m_original_threshold;
    double m_action_0_threshold;
    int m_time;
    const int m_extinction_begin;
    const int m_extinction_end;

    pcg32 m_rand;

public:
    SwitchingSlotMachineEnvironment(double p, int extinction_begin,
        int extinction_end);

    ~SwitchingSlotMachineEnvironment();

    void reset(int seed);

    double assign_reward(int action);
};

class PerArmSlotMachineEnvironment : public Environment
{
    double m_original_threshold;
    double m_action_0_threshold;
    double m_switch_likelihood;
    int m_time;
    const int m_extinction_begin;
    const int m_extinction_end;

    pcg32 m_rand;
    pcg32 m_rand_switch;

public:
    PerArmSlotMachineEnvironment(double p, double m_switch_likelihood,
        int extinction_begin, int extinction_end);

    ~PerArmSlotMachineEnvironment();

    void reset(int seed);

    double assign_reward(int action);
};

#endif // #ifndef ENVIRONMENTS_H
