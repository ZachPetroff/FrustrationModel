#ifndef BODIES_H
#define BODIES_H

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

#endif // #ifndef BODIES_H
