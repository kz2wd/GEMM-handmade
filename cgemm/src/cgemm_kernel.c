#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_kernel.h"

#include <memset.h>
#include <stdlib.h>

#include "cgemm_args.h"


#define ALIGNEMENT 64

/*
C: U*V
A: U*W
B: V*X
*/
#define U 12
#define V 16
#define W 128
#define X 128


PyObject* aligned_memory_prepare(PyObject* self, PyObject* args) {
    size_t K;
    if (!PyArg_ParseTuple(args, "n", &K)) return NULL;

    PyGEMMArgs *gemm_args;
    gemm_args = (PyGEMMArgs *) PyGEMMArgsType.tp_alloc(&PyGEMMArgsType, 0);
    
    gemm_args->A = (double *) aligned_alloc(ALIGNEMENT, sizeof(double) * K * K);
    gemm_args->B = (double *) aligned_alloc(ALIGNEMENT, sizeof(double) * K * K);
    gemm_args->C = (double *) aligned_alloc(ALIGNEMENT, sizeof(double) * K * K);
    gemm_args->K = K;

    init_mat(gemm_args->A, K);
    init_mat(gemm_args->B, K);
    memset(gemm_args->C, 0, sizeof(double) * K * K);

    return (PyObject *)gemm_args;
}


inline void block(double* sA, double* sB, double* sC) {
    for (int k = 0; k < W; ++k){
        for (int m = 0; m < sM; ++m) {
            for (int n = 0; n < sN; ++n) {
                at(C, m, n) += at(A, m, k) * at(B, k, n);
            }
        }
    }
    for (int k = 0; k < X; ++k){
        for (int m = 0; m < sM; ++m) {
            for (int n = 0; n < sN; ++n) {
                at(C, m, n) += at(A, m, k) * at(B, k, n);
            }
        }
    }
}

PyObject* kernel_compute(PyObject* self, PyObject* args) {
    
    PyGEMMArgs* gemm_args;
    
    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;
    
    double* A = gemm_args->A;
    double* B = gemm_args->B;
    double* C = gemm_args->C;
    size_t K = gemm_args->K;

    size_t sK = 
    for (size_t k = 0; k < K; ++k) {
        for (size_t m = 0; m < K; ++m) {
            for (size_t n = 0; n < K; ++n) {
                block()
            }
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}





