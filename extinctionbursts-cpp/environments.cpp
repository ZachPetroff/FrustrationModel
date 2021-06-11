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
