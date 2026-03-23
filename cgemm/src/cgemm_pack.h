#ifndef CGEMM_PACK
#define CGEMM_PACK

#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject* pack_compute(PyObject* self, PyObject* args);

typedef const double* const f64ro;
typedef double* const f64rw;
typedef const size_t dim;

void pack_compute_intern(f64ro A, dim lda, f64ro B, dim ldb, f64rw C, dim ldc, dim K);

#endif // CGEMM_PACK
