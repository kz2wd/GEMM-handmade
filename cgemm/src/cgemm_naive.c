#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_naive.h"

#define at(M, i, j) (M[i + K * j]) 

static PyObject*
cgemm_naive(PyObject* self, PyObject* args) {
    double* A;
    double* B;
    double* C;
    size_t K;

    if (!PyArg_ParseTuple(args, "ii", &a, &b)) return NULL;

    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < K; ++m) {
            for (int n = 0; n < K; ++n) {
                at(C, m, n) += at(A, m, k) * at(B, k, n);
            }
        }
    }
}

 
py_add(PyObject* self, PyObject* args){
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) return NULL;
    return PyLong_FromLong(a + b);
}
