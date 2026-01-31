#ifndef CGEMM_ARGS
#define CGEMM_ARGS

#include <Python.h>

// C side structure with python head to have a python ref
typedef struct {
    PyObject_HEAD
    double* A;
    double* B;
    double* C;
    size_t K;
} PyGEMMArgs;

void PyGEMMArgs_dealloc(PyObject* o);

// define type https://docs.python.org/3/extending/newtypes_tutorial.html
extern PyTypeObject PyGEMMArgsType;


PyObject* get_naive(PyObject* self, PyObject* args);

void init_mat(double* M, size_t K);

// i is row, j is column
// Beware of indexes order
#define at(M, i, j) (M[(K) * (i) + (j)]) 

#endif // CGEMM_ARGS