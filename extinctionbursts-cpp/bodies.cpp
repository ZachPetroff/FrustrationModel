#include "bodies.h"

Body::~Body() {}

// NullBody implementation
NullBody::~NullBody() {}

void NullBody::reset(int seed)
{
    m_total_reward = 0.;
}

double NullBody::perceive_reward(double reward)
{
    m_total_reward += reward;
    return reward;
}

double NullBody::get_fitness(int length)
{
    return m_total_reward / length;
}

InfoGainBody::InfoGainBody(int extinction_begin)
    : m_extinction_begin(extinction_begin) {}

InfoGainBody::~InfoGainBody() {}

void InfoGainBody::reset(int seed)
{
    m_total_reward = 0.;
    m_time = 0;
}

double InfoGainBody::perceive_reward(double reward) 
{
    if (m_time >= m_extinction_begin)
        m_total_reward += reward;
    m_time++;
    return reward;
}

double InfoGainBody::get_fitness(int length) 
{
    return m_total_reward / (length - m_extinction_begin);
}

NoisyBody::NoisyBody(double switch_prob)
    : m_switch_prob(uint32_t(switch_prob* (double)0x100000000)) {}

NoisyBody::~NoisyBody() {}

void NoisyBody::reset(int seed)
{
    m_total_reward = 0.;
    m_reward_access = true;
    m_rand = pcg32(seed);
}

double NoisyBody::perceive_reward(double reward)
{
    const bool switch_access = (m_rand() < m_switch_prob);
    if (switch_access) {
        m_reward_access = !m_reward_access;
    }
    if (m_reward_access) {
        m_total_reward += reward;
        return reward;
    }
    return 0.;
}

double NoisyBody::get_fitness(int length) {
    return m_total_reward / length;
}