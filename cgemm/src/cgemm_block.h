#ifndef CGEMM_BLOCK
#define CGEMM_BLOCK

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject* block_compute(PyObject* self, PyObject* args);

#endif // CGEMM_BLOCK
