#ifndef CGEMM_KERNEL
#define CGEMM_KERNEL

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject* aligned_memory_prepare(PyObject* self, PyObject* args);

PyObject* kernel_compute(PyObject* self, PyObject* args);

typedef const double* const f64ro;
typedef double* const f64rw;
typedef const size_t dim;

void kernel_compute_intern(f64ro A, f64ro B, f64rw C, dim K);

#endif // CGEMM_KERNEL
