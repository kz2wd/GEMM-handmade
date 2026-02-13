#include <math.h>
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


typedef struct {
    __m256d c00, c10;
    __m256d c01, c11;
    __m256d c02, c12;
    __m256d c03, c13;
} ckernel;



static double* prepare_ccb(dim K) {
    double* ccb = (double *) aligned_alloc(ALIGNEMENT, sizeof(double) * U * K);
    return ccb;
}

static void free_ccb(double* ccb){
    free(ccb);
}

static void sparse_copy_B(f64ro sB, f64rw csb, dim K) {
    dim Ud = U * sizeof(double);

    __builtin_prefetch(sB + 0 * K, 0, 3);
    __builtin_prefetch(sB + 1 * K, 0, 3);
    __builtin_prefetch(sB + 2 * K, 0, 3);
    __builtin_prefetch(sB + 3 * K, 0, 3);
    __builtin_prefetch(sB + 4 * K, 0, 3);
    __builtin_prefetch(sB + 5 * K, 0, 3);
    __builtin_prefetch(sB + 6 * K, 0, 3);
    __builtin_prefetch(sB + 7 * K, 0, 3);

    for (size_t w = 0; w < K; w += 8) {
        __builtin_prefetch(sB + (w + 0 + 8) * K, 0, 2);
        __builtin_prefetch(sB + (w + 1 + 8) * K, 0, 2);
        __builtin_prefetch(sB + (w + 2 + 8) * K, 0, 2);
        __builtin_prefetch(sB + (w + 3 + 8) * K, 0, 2);
        __builtin_prefetch(sB + (w + 4 + 8) * K, 0, 3);
        __builtin_prefetch(sB + (w + 5 + 8) * K, 0, 3);
        __builtin_prefetch(sB + (w + 6 + 8) * K, 0, 3);
        __builtin_prefetch(sB + (w + 7 + 8) * K, 0, 3);
        memcpy(csb + (w + 0) * U, sB + (w + 0) * K, Ud);
        memcpy(csb + (w + 1) * U, sB + (w + 1) * K, Ud);
        memcpy(csb + (w + 2) * U, sB + (w + 2) * K, Ud);
        memcpy(csb + (w + 3) * U, sB + (w + 3) * K, Ud);
        memcpy(csb + (w + 4) * U, sB + (w + 4) * K, Ud);
        memcpy(csb + (w + 5) * U, sB + (w + 5) * K, Ud);
        memcpy(csb + (w + 6) * U, sB + (w + 6) * K, Ud);
        memcpy(csb + (w + 7) * U, sB + (w + 7) * K, Ud);
    }
}


static ckernel load_kernel_block(f64rw kC, dim K){
    ckernel core;

    core.c00 = _mm256_load_pd(kC);
    core.c10 = _mm256_load_pd(kC + 4);

    core.c01 = _mm256_load_pd(kC + K);
    core.c11 = _mm256_load_pd(kC + 4 + K);

    core.c02 = _mm256_load_pd(kC + 2 * K);
    core.c12 = _mm256_load_pd(kC + 4 + 2 * K);

    core.c03 = _mm256_load_pd(kC + 3 * K);
    core.c13 = _mm256_load_pd(kC + 4 + 3 * K);

    return core;
}

static ckernel prepare_kernel_block(){
    ckernel core;

    core.c00 = _mm256_setzero_pd();
    core.c10 = _mm256_setzero_pd();

    core.c01 = _mm256_setzero_pd();
    core.c11 = _mm256_setzero_pd();

    core.c02 = _mm256_setzero_pd();
    core.c12 = _mm256_setzero_pd();

    core.c03 = _mm256_setzero_pd();
    core.c13 = _mm256_setzero_pd();

    return core;
}

static void store_kernel_block(f64rw kC, ckernel* core,  dim K){
    _mm256_store_pd(kC, core->c00);
    _mm256_store_pd(kC + 4, core->c10);

    _mm256_store_pd(kC + K, core->c01);
    _mm256_store_pd(kC + 4 + K, core->c11);

    _mm256_store_pd(kC + 2 * K, core->c02);
    _mm256_store_pd(kC + 4 + 2 * K, core->c12);

    _mm256_store_pd(kC + 3 * K, core->c03);
    _mm256_store_pd(kC + 4 + 3 * K, core->c13);

}

static void ckernel8x4_csb(f64ro srA, f64ro csb, ckernel* core, dim K){

    for (size_t w = 0; w < W; ++w) {

        __m256d b00 = _mm256_load_pd(csb + w * U);
        __m256d b10 = _mm256_load_pd(csb + 4 + w * U);

        __m256d a000 =_mm256_broadcast_sd(srA + w);
        core->c00 = _mm256_fmadd_pd(a000, b00, core->c00);
        core->c10 = _mm256_fmadd_pd(a000, b10, core->c10);

        __m256d a010 =_mm256_broadcast_sd(srA + K + w);
        core->c01 = _mm256_fmadd_pd(a010, b00, core->c01);
        core->c11 = _mm256_fmadd_pd(a010, b10, core->c11);

        __m256d a020 =_mm256_broadcast_sd(srA + 2 * K + w);
        core->c02 = _mm256_fmadd_pd(a020, b00, core->c02);
        core->c12 = _mm256_fmadd_pd(a020, b10, core->c12);

        __m256d a030 =_mm256_broadcast_sd(srA + 3 * K + w);
        core->c03 = _mm256_fmadd_pd(a030, b00, core->c03);
        core->c13 = _mm256_fmadd_pd(a030, b10, core->c13);
    }
}

// full compute of a C block of shape (U, V) that depends on row of A (K, V) and column of B (U, K)
static void c_block(f64ro rowA, f64ro colB, f64rw ccb, f64rw kC, dim K) {
    dim KW = K/W;

    // Load C into registers
    ckernel core = prepare_kernel_block();

    for (size_t kw = 0; kw < KW; ++kw) {
        // sub row of A: (W, V)
        f64ro srA = rowA + kw * W;
        // sub col of B: (X, U)
        f64ro csb = ccb + kw * W;

        // ckernel8x4(srA, scB, kC, K);
        ckernel8x4_csb(srA, csb, &core, K);
    }

    store_kernel_block(kC, &core, K);

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

    f64rw ccb = prepare_ccb(K);

    for (size_t ku = 0; ku < KU; ++ku) {

        // todo: extract B sublock so that it becomes continuous
        // Column of B: (U, K)
        f64ro colB = B + ku * U;
        sparse_copy_B(colB, ccb, K);
        for (size_t kv = 0; kv < KV; ++kv) {
            // kC: kernel size block of C: (U,V)
            f64rw kC = C + ku * U + kv * V * K;

            // Row of A: (K, V)
            f64ro rowA = A + kv * V * K;
            c_block(rowA, colB, ccb, kC, K);
        }
    }

    free_ccb(ccb);
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
