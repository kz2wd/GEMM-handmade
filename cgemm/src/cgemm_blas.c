#include <stddef.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_blas.h"

#include <cblas.h>

#include "cgemm_args.h"


void cblas_compute_intern(f64ro A, f64ro B, f64rw C, dim K){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 1.0, A, K, B, K, 0.0, C, K);
    // dim D = 512;
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A, K, B, K, 1.0, C, K);
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A + D, K, B + D * K, K, 1.0, C, K);

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A, K, B + D, K, 1.0, C + D, K);
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A + D, K, B + D + D * K, K, 1.0, C + D, K);

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A + D * K, K, B, K, 1.0, C + D * K, K);
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A + D + D * K, K, B + D * K, K, 1.0, C + D * K, K);

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A + D * K, K, B + D, K, 1.0, C + D + D * K, K);
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, D, D, 1.0, A + D + D * K, K, B + D + D * K, K, 1.0, C + D + D * K, K);
}

PyObject* cblas_compute(PyObject* self, PyObject* args) {

    PyGEMMArgs* gemm_args;

    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;

    double * A = gemm_args->A;
    double * B = gemm_args->B;
    double * C = gemm_args->C;
    const size_t K = gemm_args->K;


    cblas_compute_intern(A, B, C, K);

    Py_INCREF(Py_None);
    return Py_None;
}
