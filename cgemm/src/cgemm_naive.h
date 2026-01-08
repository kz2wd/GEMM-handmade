#ifndef CGEMM_NAIVE
#define CGEMM_NAIVE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

PyObject* naive_prepare(PyObject* self, PyObject* args);

PyObject* naive_compute(PyObject* self, PyObject* args);

PyObject* naive_debug_print(PyObject* self, PyObject* args);

#endif
