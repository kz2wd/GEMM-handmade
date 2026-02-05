#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_kernel.h"

#include <string.h>
#include <stdlib.h>

#include "cgemm_args.h"

// 64 alignement to fit full cache line
#define ALIGNEMENT 64 

/*
C: U*V
A: W*V
B: U*W

Constraints:
- U * V <= 12 * (4 * 4)
- W % U = 0 ?
- W % V = 0 ?
- W * U + W * V <= 3200 (4000 hard)
- U % 8 = 0
- V % 8 = 0
- W = 2^n

*/
#define U 128
#define V 128
#define W 128


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


//static double* kC;
//static double* sB;


// Compute sub block of shape (U,V)
static void kernel(double * kA, double * kB, double* kC, const size_t K){
    for (size_t u = 0; u < U; ++u) {
        for (size_t v = 0; v < V; ++v) {
            for (size_t w = 0; w < W; ++w) {
                kC[u * K + v] += kA[w * K + v] * kB[w + u * K];
            }
        }
    }
}


// split blocks of shape A: (W,V) and B: (U,X) into (U,V)
static void sub_block(double * sA, double * sB, double* kC, size_t K) {
    const size_t WU = W/U;
    const size_t WV = W/V;

    for (size_t wu = 0; wu < WU; ++wu) {
        double * kA = sA + wu * U * K;
        for (size_t wv = 0; wv < WV; ++wv) {
            double * kB = sB + wv * V;
            kernel(kA, kB, kC, K);
        }
    }

}

// full compute of a C block of shape (U, V) that depends on column of A (K, v) and row of B (u, K) 
static void c_block(double * colA, double * rowB, double* kC,  size_t K) {

    const size_t KW = K/W;
    // Load C into registers
    for (size_t kw = 0; kw < KW; ++kw) {
        // sub block of A: (W, V)
        double* sA = colA + kw * W * K;
        // sub block of B: (X, U)
        double* sB = rowB + kw * W;
        sub_block(sA, sB, kC, K);
        
    }
}

void kernel_compute_intern(double* A, double* B, double* C, size_t K) {
    const size_t KU = K/U;
    const size_t KV = K/V;

    const size_t KW = K/W;
    const size_t WU = W/U;
    const size_t WV = W/V;

    for (size_t ku = 0; ku < KU; ++ku) {
        for (size_t kv = 0; kv < KV; ++kv) {
            // kC: kernel size block of C: (U,V)
            double * kC = C + ku * U * K + kv * V;
            // Column of A: (K, V)
            double * colA = A + kv * V;
            // Row of B: (U, K)
            double * rowB = B + ku * U * K;
            c_block(colA, rowB, kC, K);
        }
    }
}

PyObject* kernel_compute(PyObject* self, PyObject* args) {
    
    PyGEMMArgs* gemm_args;
    
    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;
    
    double * A = gemm_args->A;
    double * B = gemm_args->B;
    double * C = gemm_args->C;
    const size_t K = gemm_args->K;

   
    kernel_compute_intern(A, B, C, K);

    Py_INCREF(Py_None);
    return Py_None;
}





