#ifndef CGEMM_NAIVE
#define CGEMM_NAIVE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>


static PyObject* cgemm_naive(double* A, double* B, double* C, size_t K);

#endif

