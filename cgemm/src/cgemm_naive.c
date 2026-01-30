#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_naive.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cgemm_args.h"


void init_mat(double* M, size_t K) {
 
    double x = RAND_MAX / 100.0;
    for (size_t m = 0; m < K; ++m) {
        for (size_t n = 0; n < K; ++n) {
            at(M, m, n) = rand() / x;
        }
    }
}

PyObject* naive_prepare(PyObject* self, PyObject* args) {
    size_t K;
    if (!PyArg_ParseTuple(args, "n", &K)) return NULL;

    PyGEMMArgs *gemm_args;
    gemm_args = (PyGEMMArgs *) PyGEMMArgsType.tp_alloc(&PyGEMMArgsType, 0);

    gemm_args->A = (double *) malloc(sizeof(double) * K * K);
    gemm_args->B = (double *) malloc(sizeof(double) * K * K);
    gemm_args->C = (double *) calloc(K * K, sizeof(double));
    gemm_args->K = K;

    init_mat(gemm_args->A, K);
    init_mat(gemm_args->B, K);

    return (PyObject *)gemm_args;
}


PyObject* naive_compute(PyObject* self, PyObject* args) {
    
    PyGEMMArgs* gemm_args;
    
    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;
    
    double* A = gemm_args->A;
    double* B = gemm_args->B;
    double* C = gemm_args->C;
    size_t K = gemm_args->K;


    for (size_t k = 0; k < K; ++k) {
        for (size_t m = 0; m < K; ++m) {
            for (size_t n = 0; n < K; ++n) {
                at(C, m, n) += at(A, m, k) * at(B, k, n);
            }
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}


void print_mat(double* M, size_t K) {
    for (size_t m = 0; m < K; ++m) {
        for (size_t n = 0; n < K; ++n) {
            printf("%f ", at(M, m, n));
        }
    printf("\n");
    }
}

PyObject* naive_debug_print(PyObject* self, PyObject* args) {
    PyGEMMArgs* gemm_args;
    
    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;
    
    double* A = gemm_args->A;
    double* B = gemm_args->B;
    double* C = gemm_args->C;
    size_t K = gemm_args->K;

    if (K > 64) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    printf("A:\n");
    print_mat(A, K);
    printf("\n\n");

    printf("B:\n");
    print_mat(B, K);
    printf("\n\n");

    printf("C:\n");
    print_mat(C, K);
    printf("\n\n");

    Py_INCREF(Py_None);
    return Py_None;
    
}


