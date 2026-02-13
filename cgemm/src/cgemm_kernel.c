#include <stddef.h>
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
#define U 8
#define V 4
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


typedef const double* const f64ro;
typedef double* const f64rw;
typedef const size_t dim;

// __m256d c00;
// __m256d c10;
// __m256d c01;
// __m256d c11;
// __m256d c02;
// __m256d c12;
// __m256d c03;
// __m256d c13;

// // Util func, should never be used
// static void load_kernel_block(f64rw kC,  dim K){
//     c00 = _mm256_load_pd(kC);
//     c10 = _mm256_load_pd(kC + 4);

//     c01 = _mm256_load_pd(kC + K);
//     c11 = _mm256_load_pd(kC + 4 + K);

//     c02 = _mm256_load_pd(kC + 2 * K);
//     c12 = _mm256_load_pd(kC + 4 + 2 * K);

//     c03 = _mm256_load_pd(kC + 3 * K);
//     c13 = _mm256_load_pd(kC + 4 + 3 * K);
// }

// static void prepare_kernel_block(){
//     c00 = _mm256_setzero_pd();
//     c10 = _mm256_setzero_pd();

//     c01 = _mm256_setzero_pd();
//     c11 = _mm256_setzero_pd();

//     c02 = _mm256_setzero_pd();
//     c12 = _mm256_setzero_pd();

//     c03 = _mm256_setzero_pd();
//     c13 = _mm256_setzero_pd();
// }

// static void store_kernel_block(f64rw kC,  dim K){
//     _mm256_store_pd(kC, c00);
//     _mm256_store_pd(kC + 4, c10);

//     _mm256_store_pd(kC + K, c01);
//     _mm256_store_pd(kC + 4 + K, c11);

//     _mm256_store_pd(kC + 2 * K, c02);
//     _mm256_store_pd(kC + 4 + 2 * K, c12);

//     _mm256_store_pd(kC + 3 * K, c03);
//     _mm256_store_pd(kC + 4 + 3 * K, c13);

// }


// kA: (U, V) @ ssB: (U, U) = kC: (U, V)
static void ckernel8x4(f64ro kA, f64ro ssB, f64rw kC, dim K){
    __m256d c00 = _mm256_load_pd(kC);
    __m256d c10 = _mm256_load_pd(kC + 4);

    __m256d c01 = _mm256_load_pd(kC + K);
    __m256d c11 = _mm256_load_pd(kC + 4 + K);

    __m256d c02 = _mm256_load_pd(kC + 2 * K);
    __m256d c12 = _mm256_load_pd(kC + 4 + 2 * K);

    __m256d c03 = _mm256_load_pd(kC + 3 * K);
    __m256d c13 = _mm256_load_pd(kC + 4 + 3 * K);
    for (size_t u = 0; u < U; ++u) {

        __m256d b00 = _mm256_load_pd(ssB + K * u);
        __m256d b10 = _mm256_load_pd(ssB + 4 + K * u);

        __m256d a000 =_mm256_broadcast_sd(kA + u);
        c00 = _mm256_fmadd_pd(a000, b00, c00);
        c10 = _mm256_fmadd_pd(a000, b10, c10);

        __m256d a010 =_mm256_broadcast_sd(kA + K + u);
        c01 = _mm256_fmadd_pd(a010, b00, c01);
        c11 = _mm256_fmadd_pd(a010, b10, c11);

        __m256d a020 =_mm256_broadcast_sd(kA + 2 * K + u);
        c02 = _mm256_fmadd_pd(a020, b00, c02);
        c12 = _mm256_fmadd_pd(a020, b10, c12);

        __m256d a030 =_mm256_broadcast_sd(kA + 3 * K + u);
        c03 = _mm256_fmadd_pd(a030, b00, c03);
        c13 = _mm256_fmadd_pd(a030, b10, c13);
    }
    _mm256_store_pd(kC, c00);
    _mm256_store_pd(kC + 4, c10);

    _mm256_store_pd(kC + K, c01);
    _mm256_store_pd(kC + 4 + K, c11);

    _mm256_store_pd(kC + 2 * K, c02);
    _mm256_store_pd(kC + 4 + 2 * K, c12);

    _mm256_store_pd(kC + 3 * K, c03);
    _mm256_store_pd(kC + 4 + 3 * K, c13);

}

// kA: (U, V) @ ssB: (U, U) = kC: (U, V)
static void tmpkernel(f64ro kA, f64ro ssB, f64rw kC, dim K){

    for (size_t u = 0; u < U; ++u){
        for (size_t v = 0; v < V; ++v){
            for (size_t inner_u = 0; inner_u < U; ++inner_u){
                kC[u * K + v] += kA[v * K + inner_u] * ssB[inner_u * K + u];
            }
        }
    }
}

// split blocks of shape sA: (W,V) and sB: (U,W) into kC (U,V)
static void sub_block(f64ro sA, f64ro sB, f64rw kC, dim K) {
    dim WU = W/U;
    dim WV = W/V;

    for (size_t wu = 0; wu < WU; ++wu) {
        f64ro kA = sA + wu * U;

        f64ro ssB = sB + wu * U * K;
        ckernel8x4(kA, ssB, kC, K);
    }

}


// full compute of a C block of shape (U, V) that depends on row of A (K, V) and column of B (U, K)
static void c_block(f64ro rowA, f64ro colB, f64rw kC, dim K) {
    dim KW = K/W;
    // Load C into registers
    for (size_t kw = 0; kw < KW; ++kw) {
        // sub block of A: (W, V)
        f64ro sA = rowA + kw * W;
        // sub block of B: (X, U)
        f64ro sB = colB + kw * W * K;
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
void kernel_compute_intern(f64ro A, f64ro B, f64rw C, dim K) {
    dim KU = K/U;
    dim KV = K/V;

    for (size_t ku = 0; ku < KU; ++ku) {
        for (size_t kv = 0; kv < KV; ++kv) {
            // kC: kernel size block of C: (U,V)
            f64rw kC = C + ku * U * K + kv * V;
            // Column of B: (U, K)
            f64ro colB = B + ku * U;
            // Row of A: (K, V)
            f64ro rowA = A + kv * V * K;
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
