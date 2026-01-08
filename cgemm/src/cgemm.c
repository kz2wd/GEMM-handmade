#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cgemm_args.h"
#include "cgemm_naive.h"
#include "cugemm_naive.h"

static int
cgemm_module_exec(PyObject* m) {

    if (PyType_Ready(&PyGEMMArgsType) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "PyGEMMArgs", (PyObject *) &PyGEMMArgsType) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot cgemm_module_slots[] = {
    {Py_mod_exec, cgemm_module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL}
};


static PyMethodDef 
Methods[] = {
    {"naive_compute", naive_compute, METH_VARARGS, "naive cgemm run"},
    {"naive_prepare", naive_prepare, METH_VARARGS, "naive cgemm preparation"},
    {"naive_debug_print", naive_debug_print, METH_VARARGS, "naive cgemm print for debug (K <= 64)"},
    {"cu_naive_compute", cu_naive_compute, METH_VARARGS, "naive cugemm run"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cgemm",
    .m_doc = "Various C implementations of GEMM",
    .m_size = 0,
    .m_methods = Methods,
    .m_slots = cgemm_module_slots
};

PyMODINIT_FUNC 
PyInit_cgemm(void) {
    return PyModuleDef_Init(&module);
}