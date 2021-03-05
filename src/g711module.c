#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <stddef.h>

#include "g711module_loader.h"
#include "g711module_saver.h"


typedef float* (*LoadFunction)(const char*, unsigned long*);
typedef bool (*SaveFunction)(const char*, const float*, unsigned long);


static inline PyObject* g711_py_load(PyObject* self, PyObject* args, PyObject *kwargs, LoadFunction load_fn);
static inline const char* parse_path(PyObject* args, PyObject *kwargs);
static inline PyArrayObject* c_to_numpy_arr(float* arr, unsigned long arr_len);

static inline PyObject* g711_py_save(PyObject* self, PyObject* args, PyObject *kwargs, SaveFunction save_fn);
static inline const char* parse_path_and_array(PyObject* args, PyObject *kwargs, PyObject** arr);


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


static PyObject*
g711_py_alaw_save(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_save(self, args, kwargs, g711_alaw_save);
}


static PyObject*
g711_py_ulaw_save(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_save(self, args, kwargs, g711_ulaw_save);
}


PyObject* g711_py_save(PyObject* self, PyObject* args, PyObject *kwargs, SaveFunction save_fn)
{
    PyObject* arr = {0};
    const char* path = parse_path_and_array(args, kwargs, &arr);
    if (!path || !arr) return NULL;

    float* arr_data = PyArray_DATA(arr);
    unsigned long arr_len = PyArray_SIZE(arr);

    bool status_ok = save_fn(path, arr_data, arr_len);
    Py_DECREF(arr);
    if (!status_ok) return NULL;
    else return PyBool_FromLong(status_ok);
}


const char* parse_path_and_array(PyObject* args, PyObject *kwargs, PyObject** arr)
{
    PyObject* path_obj = {0};
    PyObject* arr_obj = {0};

    static char *kwlist[] = {"path", "audio_data", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O", kwlist, PyUnicode_FSConverter, &path_obj, &arr_obj)) return NULL;

    if (path_obj == Py_None) {
        PyErr_SetString(PyExc_ValueError, "The function accepts only non-empty string or path-like objects.");
        return NULL;
    }

    *arr = PyArray_FROM_OTF(arr_obj, NPY_FLOAT32, NPY_IN_ARRAY);

    return PyBytes_AsString(path_obj);
}


static PyMethodDef g711Methods[] = {
    {"load_alaw", g711_py_alaw_load, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified A-Law encoded audio file."},
    {"load_ulaw", g711_py_ulaw_load, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified u-Law encoded audio file."},
    {"save_alaw", g711_py_alaw_save, METH_VARARGS | METH_KEYWORDS, "Encode to A-Law and save the specified audio array."},
    {"save_ulaw", g711_py_ulaw_save, METH_VARARGS | METH_KEYWORDS, "Encode to u-Law and save the specified audio array."},
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
