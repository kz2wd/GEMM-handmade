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
#define U 12
#define V 16
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


// Compute sub block of shape (U,V)
inline void kernel(double * kA, double * kB,  double* kC, const size_t K){
    for (size_t u = 0; u < U; ++u) {
        for (size_t v = 0; v < V; ++v) {
            for (size_t w = 0; w < W; ++w) {
                kC[u * U * K + v * V] += kA[w * K + v] * kB[v * K + w];
            }
        }
    }
}


// split blocks of shape A: (W,V) and B: (U,X) into (U,V)
inline void subblock(double * sA, double * sB, double* kC, size_t K) {
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
inline void Cblock(double * colA, double * rowB, double* kC,  size_t K) {

    const size_t KW = K/W;

    // Load C into registers
    for (size_t kw = 0; kw < KW; ++kw) {
        // sub block of A: (W, V)
        double* sA = colA + kw * W * K;

        for (size_t kw2 = 0; kw2 < KW; ++kw2) {
            // Here B will change more than A; rowB and sA should be in cache L1/L2 (remaining of colA can wait in L2/L3) 

            // sub block of B: (X, U)
            double* sB = rowB + kw2 * W;
            
            subblock(sA, sB, kC, K);
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

    const size_t KU = K/U;
    const size_t KV = K/V;
    
    for (size_t ku = 0; ku < KU; ++ku) {
        for (size_t kv = 0; kv < KV; ++kv) {
            // kC: kernel size block of C: (U,V)
            double * kC = C + ku * U * K + kv * V;

            // Column of A: (K, V)
            double * colA = A + ku * U * K + kv * V;
            // Row of B: (U, K)
            double * rowB = B + ku * U * K + kv * V;
            
            Cblock(colA, rowB, kC, K);
            
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}





