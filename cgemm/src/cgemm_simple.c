#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_args.h"
#include "cgemm_simple.h"

#define W 128

void fast_transpose(f64ro in, f64rw out, dim K) {
    for (size_t m = 0; m < K; ++m) {
        for (size_t n = 0; n < K; ++n) {
            at(out, m, n) = at(in, n, m);
        }
    }
}

// K is of unknow size, compiling options are limited
// We use W of known size
void simple_compute_intern(f64ro A, f64ro B, f64rw C, dim K) {
    dim KW = K/W;

    f64rw transposed_b = (double *) aligned_alloc(64, sizeof(double) * K * K);
    fast_transpose(B, transposed_b, K);

    for (size_t kw = 0; kw < KW; ++kw) {
        for (size_t mw = 0; mw < KW; ++mw) {
            for (size_t nw = 0; nw < KW; ++nw) {
                for (size_t mm = 0; mm < W; ++mm) {
                    for (size_t nn = 0; nn < W; ++nn) {
                        for (size_t kk = 0; kk < W; ++kk) {
                            dim k = kw * W + kk;
                            dim m = mw * W + mm;
                            dim n = nw * W + nn;
                            at(C, m, n) += at(A, m, k) * at(transposed_b, n, k);
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
