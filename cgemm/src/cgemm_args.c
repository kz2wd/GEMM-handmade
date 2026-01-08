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

