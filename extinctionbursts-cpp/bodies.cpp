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
    return m_total_reward/length;
}
