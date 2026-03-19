#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_args.h"
#include "cgemm_simple.h"

#define W 128
#define U 4
#define prefetch_D 64

static void packB(f64ro sB, f64rw csb, dim K) {
    dim Ud = U * sizeof(double);
    for (int w = K - 1; w >= 0; w -= 4) {
        __builtin_prefetch(sB + (w - 0 - prefetch_D) * K, 0, 3);
        __builtin_prefetch(sB + (w - 1 - prefetch_D) * K, 0, 3);
        __builtin_prefetch(sB + (w - 2 - prefetch_D) * K, 0, 3);
        __builtin_prefetch(sB + (w - 3 - prefetch_D) * K, 0, 3);
        memcpy(csb + (w - 0) * U, sB + (w - 0) * K, Ud);
        memcpy(csb + (w - 1) * U, sB + (w - 1) * K, Ud);
        memcpy(csb + (w - 2) * U, sB + (w - 2) * K, Ud);
        memcpy(csb + (w - 3) * U, sB + (w - 3) * K, Ud);
    }
}


// K is of unknow size, compiling options are limited
// We use W of known size
// 2 issues remaining but I think I get why I cannot be really simple
// We must work with blocks to fit into cache and also B cannot be fully transposed as it is a global operation
// And we only need locally transposed memory
// Lastly, we go through A along m k but B along n k so there is an extra axis which prevents the
// compiler from doing the kernel automatically, I believe.
void simple_compute_intern(f64ro A, f64ro B, f64rw C, dim K) {
    dim KW = K/W;
    f64rw packedB = (double *) aligned_alloc(64, sizeof(double) * W * U);

    for (size_t kw = 0; kw < KW; ++kw) {
        for (size_t mw = 0; mw < KW; ++mw) {
            for (size_t nw = 0; nw < KW; ++nw) {
                packB()

                for (size_t mm = 0; mm < W; ++mm) {
                    for (size_t nn = 0; nn < W; nn += 4) {
                        for (size_t kk = 0; kk < W; ++kk) {
                            dim k = kw * W + kk;
                            dim m = mw * W + mm;
                            dim n = nw * W + nn;
                            at(C, m, n + 0) += at(A, m, k) * packedB[kk], n + 0;
                            at(C, m, n + 1) += at(A, m, k) * packedB, n + 1;
                            at(C, m, n + 2) += at(A, m, k) * packedB, n + 2;
                            at(C, m, n + 3) += at(A, m, k) * packedB, n + 3;

                        }
                    }
                }
            }
        }
    }
}


PyObject* simple_compute(PyObject* self, PyObject* args) {

    PyGEMMArgs* gemm_args;

    if (!PyArg_ParseTuple(args, "O!", &PyGEMMArgsType, &gemm_args)) return NULL;

    double * A = gemm_args->A;
    double * B = gemm_args->B;
    double * C = gemm_args->C;
    const size_t K = gemm_args->K;


    simple_compute_intern(A, B, C, K);


    Py_INCREF(Py_None);
    return Py_None;
}
