#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_kernel.h"

#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

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
#define U 16
#define V 8
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
static void ckernel8x16(double * kA, double * kB, double* kC, const size_t K){

    __m256d a00 = _mm256_load_pd(kA);
    __m256d b00 = _mm256_load_pd(kB);
    __m256d c00 = _mm256_load_pd(kC);

    __m256d a10 = _mm256_load_pd(kA + 4);
    __m256d b10 = _mm256_load_pd(kB);
    __m256d c10 = _mm256_load_pd(kC);

    __m256d a20 = _mm256_load_pd(kA + 8);
    __m256d b20 = _mm256_load_pd(kB);
    __m256d c20 = _mm256_load_pd(kC);

    __m256d a30 = _mm256_load_pd(kA + 12);
    __m256d b30 = _mm256_load_pd(kB);
    __m256d c30 = _mm256_load_pd(kC);

    __m256d a01 = _mm256_load_pd(kA);
    __m256d b01 = _mm256_load_pd(kB);
    __m256d c01 = _mm256_load_pd(kC);

    __m256d a11 = _mm256_load_pd(kA + 4);
    __m256d b11 = _mm256_load_pd(kB);
    __m256d c11 = _mm256_load_pd(kC);

    __m256d a21 = _mm256_load_pd(kA + 8);
    __m256d b21 = _mm256_load_pd(kB);
    __m256d c21 = _mm256_load_pd(kC);

    __m256d a31 = _mm256_load_pd(kA + 8);
    __m256d b31 = _mm256_load_pd(kB);
    __m256d c31 = _mm256_load_pd(kC);
}

// kA: (U, V) @ ssB: (U, U) = kC: (U, V)
static void tmpkernel(double* kA, double* ssB, double* kC, const size_t K){
    for (size_t u = 0; u < U; ++u){
        for (size_t v = 0; v < V; ++v){
            for (size_t inner_u = 0; inner_u < U; ++inner_u){
                kC[u * K + v] += kA[v * K + inner_u] * ssB[inner_u * K + u];
            }
        }
    }
}

// split blocks of shape sA: (W,V) and sB: (U,W) into kC (U,V)
static void sub_block(double * sA, double * sB, double* kC, size_t K) {
    const size_t WU = W/U;
    const size_t WV = W/V;

    // U = 2 V ASSUMED
    for (size_t wu = 0; wu < WU; ++wu) {
        double * kA = sA + wu * U;

        double * ssB = sB + wu * U * K;
        //ckernel8x16(kA, kB, kC, K);
        tmpkernel(kA, ssB, kC, K);


    }

}

// full compute of a C block of shape (U, V) that depends on row of A (K, V) and column of B (U, K)
static void c_block(double * rowA, double * colB, double* kC,  size_t K) {

    const size_t KW = K/W;
    // Load C into registers
    for (size_t kw = 0; kw < KW; ++kw) {
        // sub block of A: (W, V)
        double* sA = rowA + kw * W;
        // sub block of B: (X, U)
        double* sB = colB + kw * W * K;
        // todo: transpose B
        sub_block(sA, sB, kC, K);

    }
}


// A @ B = C
//       j ->
//     i B B B
//   K | B B B
//    \/ B B B
// A A A C C C
// A A A C C C
// A A A C C C
void kernel_compute_intern(double* A, double* B, double* C, size_t K) {
    const size_t KU = K/U;
    const size_t KV = K/V;

    for (size_t ku = 0; ku < KU; ++ku) {
        for (size_t kv = 0; kv < KV; ++kv) {
            // kC: kernel size block of C: (U,V)
            double * kC = C + ku * U * K + kv * V;
            // Column of B: (U, K)
            double * colB = B + ku * U;
            // Row of A: (K, V)
            double * rowA = A + kv * V * K;
            c_block(rowA, colB, kC, K);
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
