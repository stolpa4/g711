#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <stddef.h>

#include "g711module_loader.h"


typedef float* (*LoadFunction)(const char*, unsigned long*);


static inline PyObject* g711_py_load(PyObject* self, PyObject* args, PyObject *kwargs, LoadFunction load_fn);
static inline const char* parse_path(PyObject* args, PyObject *kwargs);
static inline PyArrayObject* c_to_numpy_arr(float* arr, unsigned long arr_len);


static PyObject*
g711_py_alaw_load(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_load(self, args, kwargs, g711_alaw_load);
}


static PyObject*
g711_py_ulaw_load(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_load(self, args, kwargs, g711_ulaw_load);
}


PyObject*
g711_py_load(PyObject* self, PyObject* args, PyObject *kwargs, LoadFunction load_fn)
{
    const char* path = parse_path(args, kwargs);
    if (!path) return path;

    unsigned long samples_num = {0};
    float* audio_res = load_fn(path, &samples_num);
    if(!audio_res) return NULL;

    return c_to_numpy_arr(audio_res, samples_num);
}


const char* parse_path(PyObject* args, PyObject *kwargs)
{
    PyObject* path_obj = {0};

    static char *kwlist[] = {"path", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyUnicode_FSConverter, &path_obj)) return NULL;

    if (path_obj == Py_None) {
        PyErr_SetString(PyExc_ValueError, "The function accepts only non-empty string or path-like objects.");
        return NULL;
    }

    return PyBytes_AsString(path_obj);    
}


PyArrayObject* c_to_numpy_arr(float* arr, unsigned long arr_len)
{
    npy_intp dims = (npy_intp) arr_len;    
    PyArrayObject* res = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT32, arr);
    PyArray_ENABLEFLAGS(res, NPY_OWNDATA);
    
    return res;    
}


static PyMethodDef g711Methods[] = {
    {"load_alaw", g711_py_alaw_load, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified A-Law encoded audio file."},
    {"load_ulaw", g711_py_ulaw_load, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified u-Law encoded audio file."},
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
