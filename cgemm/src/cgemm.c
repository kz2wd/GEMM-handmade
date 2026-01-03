#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_naive.h"

static PyObject* 
py_add(PyObject* self, PyObject* args){
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) return NULL;
    return PyLong_FromLong(a + b);
}



static PyMethodDef 
Methods[] = {
    {"add", py_add, METH_VARARGS, "Add two integers"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "cgemm",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC 
PyInit_cgemm(void) {
    return PyModule_Create(&module);
}