#ifndef CUGEMM_NAIVE
#define CUGEMM_NAIVE

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

PyObject* cu_naive_compute(PyObject* self, PyObject* args);

#ifdef __cplusplus
}
#endif

#endif // CUGEMM_NAIVE