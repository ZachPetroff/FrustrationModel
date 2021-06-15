#include "environments.h"

Environment::~Environment() {}

// TrueExtinctionEnvironment implementation
TrueExtinctionEnvironment::TrueExtinctionEnvironment(double p,
    int extinction_begin, int extinction_end) :
        m_always_action_0(p >= 1.),
        m_action_0_threshold(uint32_t(p*(double)0x100000000)),
        m_extinction_begin(extinction_begin),
        m_extinction_end(extinction_end) {}

TrueExtinctionEnvironment::~TrueExtinctionEnvironment() {}

void TrueExtinctionEnvironment::reset(int seed)
{
    m_rand = pcg32(seed);
    m_time = 0;
}

double TrueExtinctionEnvironment::assign_reward(int action)
{
    if(m_time >= m_extinction_begin && m_time < m_extinction_end)
    {
        m_time++;
        return 0.;
    }
    else
    {
        m_time++;
        const bool is_action_1 = (action == 1);
        if(m_always_action_0)
        {
            if(is_action_1)
                return 0.;
            else
                return 1.;
        }
        const bool action_0_wins = (m_rand() < m_action_0_threshold);
        if(action_0_wins ^ is_action_1)
            return 1.;
        else
            return 0.;
    }
}

// PerArmSlotMachineEnvironment Implementation
PerArmSlotMachineEnvironment::PerArmSlotMachineEnvironment(double p, double switch_likelihood,
    int extinction_begin, int extinction_end) :
    m_original_threshold(p),
    m_switch_likelihood(switch_likelihood),
    m_extinction_begin(extinction_begin),
    m_extinction_end(extinction_end) {}

PerArmSlotMachineEnvironment::~PerArmSlotMachineEnvironment() {}

void PerArmSlotMachineEnvironment::reset(int seed)
{
    m_rand = pcg32(seed);
    m_rand_switch = pcg32(seed);
    m_time = 0;
    m_action_0_threshold = m_original_threshold;
}

double PerArmSlotMachineEnvironment::assign_reward(int action)
{
    if (m_time >= m_extinction_begin && m_time < m_extinction_end)
    {   
        double rand_switch = (double)m_rand_switch() / (double)0x100000000;
        const bool do_switch = (rand_switch < m_switch_likelihood);
        m_action_0_threshold = do_switch ? 1 - m_action_0_threshold : m_action_0_threshold;
    }
    if (m_time == m_extinction_end) {
        m_action_0_threshold = (m_action_0_threshold < .5) ? 1 - m_action_0_threshold : m_action_0_threshold;
    }
    m_time++;
    const bool is_action_1 = (action == 1);
    double rand = (double)m_rand() / (double)0x100000000;
    const bool action_0_wins = (rand < m_action_0_threshold);
    if (action_0_wins ^ is_action_1)
        return 1.;
    else
        return 0.;

}


// SwitchingSlotMachineEnvironment Implementation
SwitchingSlotMachineEnvironment::SwitchingSlotMachineEnvironment(double p,
    int extinction_begin, int extinction_end) :
    m_original_threshold(p),
    m_extinction_begin(extinction_begin),
    m_extinction_end(extinction_end) 
    {}

SwitchingSlotMachineEnvironment::~SwitchingSlotMachineEnvironment() {}

void SwitchingSlotMachineEnvironment::reset(int seed)
{
    m_rand = pcg32(seed);
    m_time = 0;
    m_action_0_threshold = m_original_threshold;
}

double SwitchingSlotMachineEnvironment::assign_reward(int action)
{
    m_action_0_threshold = (m_time == m_extinction_begin || m_extinction_end == m_time) ? 1 - m_action_0_threshold : m_action_0_threshold;
    m_time++;
    const bool is_action_1 = (action == 1);
    double rand = (double)m_rand() / (double)0x100000000;
    const bool action_0_wins = (rand < m_action_0_threshold);
    if (action_0_wins ^ is_action_1)
        return 1.;
    else
        return 0.;
}


