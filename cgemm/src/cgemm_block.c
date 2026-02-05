#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_block.h"

#include <string.h>
#include <stdlib.h>

#include "cgemm_args.h"


#define U 128
#define V 128


// full compute of a C block of shape (U, V) that depends on column of A (K, v) and row of B (u, K) 
static void c_block(double * sA, double * sB, double* sC,  size_t K) {

    for (size_t k = 0; k < K; ++k) { 
        for (size_t u = 0; u < U; ++u) {
            for (size_t v = 0; v < V; ++v) {
                sC[u * K + v] += sA[k * K + v] * sB[k + u * K];
            }
        }
    }

}

void block_compute_intern(double* A, double* B, double* C, size_t K) {
    const size_t KU = K/U;
    const size_t KV = K/V;


    for (size_t ku = 0; ku < KU; ++ku) {
        for (size_t kv = 0; kv < KV; ++kv) {

            double * sC = C + ku * U * K + kv * V;
            // Column of A: (K, V)
            double * sA = A + ku * U * K + kv * V;
            // Row of B: (U, K)
            double * sB = B + ku * U * K + kv * V;
            c_block(sA, sB, sC, K);
        }
    }
}

PyObject* block_compute(PyObject* self, PyObject* args) {
    
    PyGEMMArgs* gemm_args;
    
    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;
    
    double * A = gemm_args->A;
    double * B = gemm_args->B;
    double * C = gemm_args->C;
    const size_t K = gemm_args->K;

   
    block_compute_intern(A, B, C, K);

    Py_INCREF(Py_None);
    return Py_None;
}





