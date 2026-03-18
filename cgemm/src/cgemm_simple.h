#ifndef CGEMM_SIMPLE
#define CGEMM_SIMPLE

#define PY_SSIZE_T_CLEAN
#include <Python.h>


PyObject* simple_compute(PyObject* self, PyObject* args);

typedef const double* const f64ro;
typedef double* const f64rw;
typedef const size_t dim;

void simple_compute_intern(f64ro A, f64ro B, f64rw C, dim K);

#endif // CGEMM_SIMPLE
