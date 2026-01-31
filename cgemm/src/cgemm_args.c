#include "cgemm_args.h"


PyTypeObject PyGEMMArgsType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cgemm.PyGEMMArgs",
    .tp_doc = PyDoc_STR("Arguments object for GEMM"),
    .tp_basicsize = sizeof(PyGEMMArgs),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = PyGEMMArgs_dealloc,
};


void PyGEMMArgs_dealloc(PyObject* o){
    PyGEMMArgs* self = (PyGEMMArgs*) o;
    free(self->A);
    free(self->B);
    free(self->C);
    Py_TYPE(self)->tp_free(self);
}


PyObject* get_naive(PyObject* self, PyObject* args) {

    PyGEMMArgs* gemm_args;
    int matrix_index; 
    int i;
    int j;
    
    if (!PyArg_ParseTuple(args, "O!iii", &PyGEMMArgsType, &gemm_args, &matrix_index, &i, &j)) return NULL;

    double* target = gemm_args->A;
    if (matrix_index == 1){
        target = gemm_args->B;
    } else if (matrix_index == 2) {
        target = gemm_args->C;
    }
    int K = gemm_args->K;
    return PyFloat_FromDouble(at(target, i, j));
}

void init_mat(double* M, size_t K) {
 
    double x = RAND_MAX / 100.0;
    for (size_t m = 0; m < K; ++m) {
        for (size_t n = 0; n < K; ++n) {
            at(M, m, n) = rand() / x;
        }
    }
}