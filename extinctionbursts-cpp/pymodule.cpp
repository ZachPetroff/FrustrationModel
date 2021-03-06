#define NPY_NO_DEPRECATED_API 1
#include <Python.h>
#include <numpy/arrayobject.h>

#include "agents.h"
#include "bodies.h"
#include "environments.h"
#include "simulate.h"

void destroy_Agent(PyObject* capsule)
{
    Agent* agent = (Agent*)PyCapsule_GetPointer(capsule, "agent");
    if (agent)
        delete agent;
}

static PyObject* make_FrustrationModelAgent(PyObject* self, PyObject* args)
{
    double reward, cost, expectation_growth, expectation_decay, temperature;

    temperature = 1.0;

    if (!PyArg_ParseTuple(args, "dddd|d", &reward, &cost, &expectation_growth, &expectation_decay, &temperature))
    {
        return nullptr;
    }

    try
    {
        FrustrationModelAgent* agent = new FrustrationModelAgent(reward, cost, expectation_growth, expectation_decay, temperature);
        return PyCapsule_New(agent, "agent", &destroy_Agent);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_UncertaintyModelAgent(PyObject* self, PyObject* args)
{
    double w_uncertainty, fast_lambda, slow_lambda, temperature;

    temperature = 1.0;

    if (!PyArg_ParseTuple(args, "ddd|d", &w_uncertainty, &fast_lambda, &slow_lambda, &temperature))
    {
        return nullptr;
    }

    try
    {
        UncertaintyModelAgent* agent = new UncertaintyModelAgent(w_uncertainty, fast_lambda, slow_lambda, temperature);
        return PyCapsule_New(agent, "agent", &destroy_Agent);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_UncertaintyModelAgent2(PyObject* self, PyObject* args)
{
    double w_uncertainty, reward_lambda, change_lambda, temperature;

    temperature = 1.0;

    if (!PyArg_ParseTuple(args, "ddd|d", &w_uncertainty, &reward_lambda, &change_lambda, &temperature))
    {
        return nullptr;
    }

    try
    {
        UncertaintyModelAgent2* agent = new UncertaintyModelAgent2(w_uncertainty, reward_lambda, change_lambda, temperature);
        return PyCapsule_New(agent, "agent", &destroy_Agent);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}
void destroy_Environment(PyObject* capsule)
{
    Environment* environment = (Environment*)PyCapsule_GetPointer(capsule, "environment");
    if (environment)
        delete environment;
}

static PyObject* make_TrueExtinctionEnvironment(PyObject* self, PyObject* args)
{
    double p;
    int extinction_begin;
    int extinction_end;

    if (!PyArg_ParseTuple(args, "dii", &p, &extinction_begin, &extinction_end))
    {
        return nullptr;
    }

    try
    {
        TrueExtinctionEnvironment* environment = new TrueExtinctionEnvironment(p, extinction_begin, extinction_end);
        return PyCapsule_New(environment, "environment", &destroy_Environment);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_SwitchingSlotMachineEnvironment(PyObject* self, PyObject* args)
{
    double p;
    int extinction_begin;
    int extinction_end;

    if (!PyArg_ParseTuple(args, "dii", &p, &extinction_begin, &extinction_end))
    {
        return nullptr;
    }

    try
    {
        SwitchingSlotMachineEnvironment* environment = new SwitchingSlotMachineEnvironment(p, extinction_begin, extinction_end);
        return PyCapsule_New(environment, "environment", &destroy_Environment);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_PerArmSlotMachineEnvironment(PyObject* self, PyObject* args)
{
    double p;
    double switch_likelihood;
    int extinction_begin;
    int extinction_end;

    if (!PyArg_ParseTuple(args, "ddii", &p, &switch_likelihood, &extinction_begin, &extinction_end))
    {
        return nullptr;
    }

    try
    {
        PerArmSlotMachineEnvironment* environment = new PerArmSlotMachineEnvironment(p, switch_likelihood, extinction_begin, extinction_end);
        return PyCapsule_New(environment, "environment", &destroy_Environment);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_DetMultiscaleEnvironment(PyObject* self, PyObject* args)
{
    double p;
    int t_switch_short, t_return_short, t_switch_long, t_return_long;

    if (!PyArg_ParseTuple(args, "diiii", &p, &t_switch_short, &t_return_short, &t_switch_long, &t_return_long))
    {
        return nullptr;
    }

    try
    {
        DetMultiscaleEnvironment* environment = new DetMultiscaleEnvironment(p, t_switch_short, t_return_short, t_switch_long, t_return_long);
        return PyCapsule_New(environment, "environment", &destroy_Environment);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_StochMultiscaleEnvironment(PyObject* self, PyObject* args)
{
    double p, p_switch_short, p_return_short, p_switch_long, p_return_long;

    if (!PyArg_ParseTuple(args, "ddddd", &p, &p_switch_short, &p_return_short, &p_switch_long, &p_return_long))
    {
        return nullptr;
    }

    try
    {
        StochMultiscaleEnvironment* environment = new StochMultiscaleEnvironment(p, p_switch_short, p_return_short, p_switch_long, p_return_long);
        return PyCapsule_New(environment, "environment", &destroy_Environment);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

void destroy_Body(PyObject* capsule)
{
    Body* body = (Body*)PyCapsule_GetPointer(capsule, "body");
    if (body)
        delete body;
}

static PyObject* make_NullBody(PyObject* self, PyObject* args)
{
    if (!PyArg_ParseTuple(args, ""))
    {
        return nullptr;
    }

    try
    {
        NullBody* body = new NullBody();
        return PyCapsule_New(body, "body", &destroy_Body);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* make_InfoGainBody(PyObject* self, PyObject* args)
{
    int extinction_begin;

    if (!PyArg_ParseTuple(args, "i", &extinction_begin))
    {
        return nullptr;
    }

    try
    {
        InfoGainBody* body = new InfoGainBody(extinction_begin);
        return PyCapsule_New(body, "body", &destroy_Body);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}


static PyObject* make_NoisyBody(PyObject* self, PyObject* args)
{
    double switch_prob;

    if (!PyArg_ParseTuple(args, "d", &switch_prob))
    {
        return nullptr;
    }

    try
    {
        NoisyBody* body = new NoisyBody(switch_prob);
        return PyCapsule_New(body, "body", &destroy_Body);
    }
    catch (std::bad_alloc& exception)
    {
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate requested class in memory.");
        return nullptr;
    }
}

static PyObject* py_simulate(PyObject* self, PyObject* args)
{
    int n, duration, seed;
    PyObject* py_agent = nullptr;
    PyObject* py_body = nullptr;
    PyObject* py_environment = nullptr;

    if (!PyArg_ParseTuple(args, "iiiOOO",
        &n, &duration, &seed, &py_agent, &py_body, &py_environment))
    {
        return nullptr;
    }

    Agent* agent = (Agent*)PyCapsule_GetPointer(py_agent, "agent");
    Body* body = (Body*)PyCapsule_GetPointer(py_body, "body");
    Environment* environment
        = (Environment*)PyCapsule_GetPointer(py_environment, "environment");

    if (!agent)
    {
        // set TypeError...
        return nullptr;
    }

    if (!body)
    {
        // set TypeError...
        return nullptr;
    }

    if (!environment)
    {
        // set TypeError...
        return nullptr;
    }

    npy_intp dim_out[1] = {(npy_intp)duration};

    PyObject* action_array = PyArray_SimpleNew(1, &dim_out[0], NPY_DOUBLE);
    PyObject* slow_expectation_0_array = PyArray_SimpleNew(1, &dim_out[0], NPY_DOUBLE);
    PyObject* slow_expectation_1_array = PyArray_SimpleNew(1, &dim_out[0], NPY_DOUBLE);
    PyObject* fast_expectation_0_array = PyArray_SimpleNew(1, &dim_out[0], NPY_DOUBLE);
    PyObject* fast_expectation_1_array = PyArray_SimpleNew(1, &dim_out[0], NPY_DOUBLE);

    /// MEMORY ERROR

    if (!action_array || !slow_expectation_0_array || !slow_expectation_1_array
        || !fast_expectation_0_array || !fast_expectation_1_array)
    {
        Py_XDECREF(action_array);
        Py_XDECREF(slow_expectation_0_array);
        Py_XDECREF(slow_expectation_1_array);
        Py_XDECREF(fast_expectation_0_array);
        Py_XDECREF(fast_expectation_1_array);
        PyErr_SetString(PyExc_MemoryError,
            "Could not allocate actions array in memory.");
        return nullptr;
    }

    SimulationResult result;
    result.action_freq = (double*)PyArray_DATA(action_array);
    result.slow_expectation_0 = (double*)PyArray_DATA(slow_expectation_0_array);
    result.slow_expectation_1 = (double*)PyArray_DATA(slow_expectation_1_array);
    result.fast_expectation_0 = (double*)PyArray_DATA(fast_expectation_0_array);
    result.fast_expectation_1 = (double*)PyArray_DATA(fast_expectation_1_array);

    simulate(n, duration, seed, *agent, *body, *environment, result);

    return Py_BuildValue("(dNNNNN)", result.fitness, action_array,
        slow_expectation_0_array,
        slow_expectation_1_array,
        fast_expectation_0_array,
        fast_expectation_1_array);
}

static PyMethodDef extinctionbursts_Methods[] = {
    {"FrustrationModelAgent", make_FrustrationModelAgent, METH_VARARGS, "Initialize a FrustrationModelAgent class."},
    {"UncertaintyModelAgent", make_UncertaintyModelAgent, METH_VARARGS, "Initialize a UncertaintyModelAgent class."},
    {"UncertaintyModelAgent2", make_UncertaintyModelAgent2, METH_VARARGS, "Initialize a UncertaintyModelAgent2 class."},
    {"TrueExtinctionEnvironment", make_TrueExtinctionEnvironment, METH_VARARGS, "Initialize a TrueExtinctionEnvironment class."},
    {"SwitchingSlotMachineEnvironment", make_SwitchingSlotMachineEnvironment, METH_VARARGS, "Initialize a SwitchingSlotMachineEnvironment class."},
    {"PerArmSlotMachineEnvironment", make_PerArmSlotMachineEnvironment, METH_VARARGS, "Initialize a PerArmSlotMachineEnvironment class."},
    {"DetMultiscaleEnvironment", make_DetMultiscaleEnvironment, METH_VARARGS, "Initialize a DetMultiscaleEnvironment class."},
    {"StochMultiscaleEnvironment", make_StochMultiscaleEnvironment, METH_VARARGS, "Initialize a StochMultiscaleEnvironment class."},
    {"NullBody", make_NullBody, METH_VARARGS, "Initialize a NullBody class."},
    {"InfoGainBody", make_InfoGainBody, METH_VARARGS, "Initialize an InfoGainBody class."},
    {"NoisyBody", make_NoisyBody, METH_VARARGS, "Initialize a NoisyBody class."},
    {"simulate", py_simulate, METH_VARARGS, "Simulate the actions that <agent> takes with <body> in <environment> for <n> runs with duration <duration> and seed <seed>. Returns the agent's fitness and a numpy array of the actions it took."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef extinctionbursts_module = {
    PyModuleDef_HEAD_INIT,
    "extinctionbursts",
    NULL,
    -1,
    extinctionbursts_Methods
};

extern "C"
{
PyMODINIT_FUNC PyInit_extinctionbursts(void) {
    import_array();
    return PyModule_Create(&extinctionbursts_module);
}
}
