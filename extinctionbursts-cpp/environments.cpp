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


// DetMultiscaleEnvironment Implementation
DetMultiscaleEnvironment::DetMultiscaleEnvironment(double p, int t_switch_short,
        int t_return_short, int t_switch_long, int t_return_long) :
    m_action_0_threshold(uint32_t(p*(double)0x100000000)),
    m_t_switch_short(t_switch_short),
    m_t_return_short(t_return_short),
    m_cycle_short(t_switch_short+t_return_short),
    m_t_switch_long(t_switch_long),
    m_t_return_long(t_return_long),
    m_cycle_long(t_switch_long+t_return_long)
    {}

DetMultiscaleEnvironment::~DetMultiscaleEnvironment() {}

void DetMultiscaleEnvironment::reset(int seed)
{
    m_rand = pcg32(seed);
    if(m_switched_short ^ m_switched_long)
        m_action_0_threshold = ~m_action_0_threshold;
    m_time = 0;
    m_switched_short = false;
    m_switched_long = false;
}

double DetMultiscaleEnvironment::assign_reward(int action)
{
    m_time ++;
    if(m_time % m_cycle_short == 0 || m_time % m_cycle_short == m_t_switch_short)
    {
        m_switched_short = !m_switched_short;
        m_action_0_threshold = ~m_action_0_threshold;
    }
    if(m_time % m_cycle_long == 0 || m_time % m_cycle_long == m_t_switch_long)
    {
        m_switched_long = !m_switched_long;
        m_action_0_threshold = ~m_action_0_threshold;
    }
    const bool is_action_1 = (action == 1);
    const bool action_0_wins = (m_rand() < m_action_0_threshold);
    if (action_0_wins ^ is_action_1)
        return 1.;
    else
        return 0.;
}


// StochMultiscaleEnvironment Implementation
StochMultiscaleEnvironment::StochMultiscaleEnvironment(double p,
        double p_switch_short, double p_return_short, double p_switch_long,
        double p_return_long) :
    m_action_0_threshold(uint32_t(p*(double)0x100000000)),
    m_switch_short_threshold(uint32_t(p_switch_short*(double)0x100000000)),
    m_return_short_threshold(uint32_t(p_return_short*(double)0x100000000)),
    m_switch_long_threshold(uint32_t(p_switch_long*(double)0x100000000)),
    m_return_long_threshold(uint32_t(p_return_long*(double)0x100000000))
    {}

StochMultiscaleEnvironment::~StochMultiscaleEnvironment() {}

void StochMultiscaleEnvironment::reset(int seed)
{
    m_rand = pcg32(seed);
    if(m_switched_short ^ m_switched_long)
        m_action_0_threshold = ~m_action_0_threshold;
    m_switched_short = false;
    m_switched_long = false;
}

double StochMultiscaleEnvironment::assign_reward(int action)
{
    if(m_switched_short)
    {
        if(m_rand() < m_return_short_threshold)
        {
            m_switched_short = false;
            m_action_0_threshold = ~m_action_0_threshold;
        }
    }
    else
    {
        if(m_rand() < m_switch_short_threshold)
        {
            m_switched_short = true;
            m_action_0_threshold = ~m_action_0_threshold;
        }
    }

    if(m_switched_long)
    {
        if(m_rand() < m_return_long_threshold)
        {
            m_switched_long = false;
            m_action_0_threshold = ~m_action_0_threshold;
        }
    }
    else
    {
        if(m_rand() < m_switch_long_threshold)
        {
            m_switched_long = true;
            m_action_0_threshold = ~m_action_0_threshold;
        }
    }

    const bool is_action_1 = (action == 1);
    const bool action_0_wins = (m_rand() < m_action_0_threshold);
    if (action_0_wins ^ is_action_1)
        return 1.;
    else
        return 0.;
}


