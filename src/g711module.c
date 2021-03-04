#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <stddef.h>

#include "g711module_loader.h"


static PyObject*
g711_py_load(PyObject* self, PyObject* args)
{
    const char* path = {0};
    PyObject* path_obj = {0};

    if (!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &path_obj)) return NULL;

    if (path_obj == Py_None) {
        PyErr_SetString(PyExc_ValueError, "The function accepts only non-empty string or path-like objects.");
        return NULL;
    }

    path = PyBytes_AsString(path_obj);

    unsigned long samples_num = {0};
    float* audio_res = g711_alaw_load(path, &samples_num);

    if(!audio_res) return NULL;

    npy_intp dims = (npy_intp) samples_num;
    PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT32, audio_res);
    PyArrayObject* res = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT32, audio_res);
    PyArray_ENABLEFLAGS(res, NPY_OWNDATA);
    return (PyObject *)res;
}


static PyMethodDef g711Methods[] = {
    {"load", g711_py_load, METH_VARARGS, "Load and decode the specified A-Law encoded audio file."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef g711module = {
    PyModuleDef_HEAD_INIT,
    "g711",
    "A small library load and save A-Law encoded audio files.",
    -1,
    g711Methods
};


PyMODINIT_FUNC PyInit_g711(void) {
    import_array();
    return PyModule_Create(&g711module);
}
