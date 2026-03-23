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
#define Q 256
#define P 256

#define U 8
#define V 4

void pack_compute_intern(f64ro A, dim lda, f64ro B, dim ldb, f64rw C, dim ldc, dim K) {
    dim KO = K / O;
    dim KP = K / P;
    dim KQ = K / Q;

    for (int ko = 0; ko < KO; ++ko) {
        for (int kp = 0; kp < KP; ++kp) {
            for (int kq = 0; kq < KQ; ++kq) {
                f64ro sA = A + kq * Q + ko * O * lda;
                f64ro sB = B + kp * P + kq * Q * ldb;
                f64ro sC = C + kp * P + ko * O * ldc;

                f64ro pA = packA();
                f64ro pB = packB();

                compute_pack()
            }
        }
    }

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
