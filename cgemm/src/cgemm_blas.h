#ifndef CGEMM_CBLAS
#define CGEMM_CBLAS

#define PY_SSIZE_T_CLEAN
#include <Python.h>


PyObject* cblas_compute(PyObject* self, PyObject* args);

typedef const double* const f64ro;
typedef double* const f64rw;
typedef const size_t dim;

void cblas_compute_intern(f64ro A, f64ro B, f64rw C, dim K);

#endif // CGEMM_CBLAS
