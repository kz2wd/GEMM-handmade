#include <stddef.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_pack.h"

#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

#include "cgemm_args.h"

// 64 alignement to fit full cache line
#define ALIGNEMENT 64

// Prefetech distance
#define prefetch_D 64

#define O 256
#define Q 32
#define P 256

#define U 8
#define V 4

static f64rw allocate_pa() {
    return (f64rw) aligned_alloc(ALIGNEMENT, sizeof(double) * O * Q);
}

static f64rw allocate_pb() {
    return (f64rw) aligned_alloc(ALIGNEMENT, sizeof(double) * Q * P);
}

static void packA(f64rw pa, f64ro sA, dim lda) {
    dim OV = O / V;
    for (int ov = 0; ov < OV; ++ov) {
        for (int q = 0; q < Q; ++q) {
            for (int v = 0; v < V; ++v) {
                pa[ov * V * Q + v + q * V] = sA[(ov * V + v) * lda + q];
            }
        }
    }
}

static void packB(f64rw pb, f64ro sB, dim ldb) {
    dim PU = P / U;
    dim Ud = sizeof(double) * U;
    for (int pu = 0; pu < PU; ++pu) {
        for (int q = 0; q < Q; ++q) {
            // pu + X prefetch so that X * Q is used as prefetch distance
            // __builtin_prefetch(sB + (pu + 4) * U + q * ldb, 0, 0); // I think it is useless ...
            memcpy(pb + pu * U * Q + q * U, sB + pu * U + q * ldb, Ud);
        }
    }
}

typedef struct {
    __m256d c00, c10;
    __m256d c01, c11;
    __m256d c02, c12;
    __m256d c03, c13;
} ckernel;

static ckernel load_kc(f64rw kC, dim ldc){
    ckernel core;

    core.c00 = _mm256_load_pd(kC + 0 + 0 * ldc);
    core.c10 = _mm256_load_pd(kC + 4 + 0 * ldc);

    core.c01 = _mm256_load_pd(kC + 0 + 1 * ldc);
    core.c11 = _mm256_load_pd(kC + 4 + 1 * ldc);

    core.c02 = _mm256_load_pd(kC + 0 + 2 * ldc);
    core.c12 = _mm256_load_pd(kC + 4 + 2 * ldc);

    core.c03 = _mm256_load_pd(kC + 0 + 3 * ldc);
    core.c13 = _mm256_load_pd(kC + 4 + 3 * ldc);

    return core;
}

static void store_kc(f64rw kC, ckernel* core, dim ldc){
    _mm256_store_pd(kC + 0 + 0 * ldc, core->c00);
    _mm256_store_pd(kC + 4 + 0 * ldc, core->c10);

    _mm256_store_pd(kC + 0 + 1 * ldc, core->c01);
    _mm256_store_pd(kC + 4 + 1 * ldc, core->c11);

    _mm256_store_pd(kC + 0 + 2 * ldc, core->c02);
    _mm256_store_pd(kC + 4 + 2 * ldc, core->c12);

    _mm256_store_pd(kC + 0 + 3 * ldc, core->c03);
    _mm256_store_pd(kC + 4 + 3 * ldc, core->c13);
}

static void compute_kernel(f64ro ra, f64ro cb, f64rw kC, dim ldc) {
    ckernel core = load_kc(kC, ldc);

    for (size_t q = 0; q < Q; ++q) {

        // Each step computes U values of cb and V values of ra
        __m256d b00 = _mm256_load_pd(cb + q * U + 0);
        __m256d b10 = _mm256_load_pd(cb + q * U + 4);

        __m256d a000 =_mm256_broadcast_sd(ra + q * V + 0);
        core.c00 = _mm256_fmadd_pd(a000, b00, core.c00);
        core.c10 = _mm256_fmadd_pd(a000, b10, core.c10);

        __m256d a010 =_mm256_broadcast_sd(ra + q * V + 1);
        core.c01 = _mm256_fmadd_pd(a010, b00, core.c01);
        core.c11 = _mm256_fmadd_pd(a010, b10, core.c11);

        __m256d a020 =_mm256_broadcast_sd(ra + q * V + 2);
        core.c02 = _mm256_fmadd_pd(a020, b00, core.c02);
        core.c12 = _mm256_fmadd_pd(a020, b10, core.c12);

        __m256d a030 =_mm256_broadcast_sd(ra + q * V + 3);
        core.c03 = _mm256_fmadd_pd(a030, b00, core.c03);
        core.c13 = _mm256_fmadd_pd(a030, b10, core.c13);
    }
    store_kc(kC, &core, ldc);
}


static void compute_pack(f64ro pa, f64ro pb, f64rw sC, dim ldc) {
    dim OV = O / V;
    dim PU = P / U;

    for (int ov = 0; ov < OV; ++ov) {
        for (int pu = 0; pu < PU; ++pu) {
            f64ro ra = pa + Q * V * ov;
            f64ro cb = pb + Q * U * pu;
            f64rw kC = sC + U * pu + ov * V * ldc;
            compute_kernel(ra, cb, kC, ldc);
        }
    }
}

void pack_compute_intern(f64ro A, dim lda, f64ro B, dim ldb, f64rw C, dim ldc, dim K) {
    dim KO = K / O;
    dim KP = K / P;
    dim KQ = K / Q;

    f64rw pa = allocate_pa();
    f64rw pb = allocate_pb();

    for (int ko = 0; ko < KO; ++ko) {
        for (int kp = 0; kp < KP; ++kp) {
            for (int kq = 0; kq < KQ; ++kq) {

                f64ro sA = A + kq * Q + ko * O * lda;
                f64ro sB = B + kp * P + kq * Q * ldb;
                f64rw sC = C + kp * P + ko * O * ldc;

                packA(pa, sA, lda);
                packB(pb, sB, ldb);

                compute_pack(pa, pb, sC, ldc);
            }
        }
    }

    free(pa);
    free(pb);

}

PyObject* pack_compute(PyObject* self, PyObject* args) {

    PyGEMMArgs* gemm_args;

    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;

    double * A = gemm_args->A;
    double * B = gemm_args->B;
    double * C = gemm_args->C;
    const size_t K = gemm_args->K;

    pack_compute_intern(A, K, B, K, C, K, K);

    Py_INCREF(Py_None);
    return Py_None;
}
