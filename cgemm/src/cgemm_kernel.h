#ifndef CGEMM_KERNEL
#define CGEMM_KERNEL

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject* aligned_memory_prepare(PyObject* self, PyObject* args);

PyObject* kernel_compute(PyObject* self, PyObject* args);

#endif // CGEMM_KERNEL
