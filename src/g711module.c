#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <stddef.h>

#include "g711module_loader.h"
#include "g711module_saver.h"

#include "g711module_coder.h"


typedef float* (*LoadFunction)(const char*, unsigned long*);
typedef bool (*SaveFunction)(const char*, const float*, unsigned long);


static inline PyObject* g711_py_load(PyObject* self, PyObject* args, PyObject *kwargs, LoadFunction load_fn);
static inline const char* parse_path(PyObject* args, PyObject *kwargs);
static inline PyObject* c_to_numpy_arr(float* arr, unsigned long arr_len);

static inline PyObject* g711_py_save(PyObject* self, PyObject* args, PyObject *kwargs, SaveFunction save_fn);
static inline const char* parse_path_and_array(PyObject* args, PyObject *kwargs, PyObject** arr);

static inline PyObject* g711_py_decode(PyObject* self, PyObject* args, PyObject *kwargs, DecodeFunction decode_fn);
static inline const char* parse_bytes(PyObject* args, PyObject *kwargs, unsigned long* bytes_number);

static inline PyObject* g711_py_encode(PyObject* self, PyObject* args, PyObject *kwargs, EncodeFunction encode_fn);
static inline PyObject* parse_array(PyObject* args, PyObject *kwargs);

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
    if (!path) return NULL;

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


PyObject* c_to_numpy_arr(float* arr, unsigned long arr_len)
{
    npy_intp dims = (npy_intp) arr_len;
    PyObject* res = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT32, arr);
    PyArray_ENABLEFLAGS((PyArrayObject*)res, NPY_ARRAY_OWNDATA);

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

    float* arr_data = PyArray_DATA((PyArrayObject *)arr);
    unsigned long arr_len = PyArray_SIZE((PyArrayObject *)arr);

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

    *arr = PyArray_FROM_OTF(arr_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    return PyBytes_AsString(path_obj);
}


static PyObject*
g711_py_alaw_decode(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_decode(self, args, kwargs, g711_alaw_decode);
}


static PyObject*
g711_py_ulaw_decode(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_decode(self, args, kwargs, g711_ulaw_decode);
}


PyObject* g711_py_decode(PyObject* self, PyObject* args, PyObject *kwargs, DecodeFunction decode_fn)
{
    unsigned long bts_num = {0};
    const char* bts = parse_bytes(args, kwargs, &bts_num);
    if (!bts) return NULL;

    float* audio_res = g711_decode(bts, bts_num, decode_fn);
    if(!audio_res) return NULL;

    return c_to_numpy_arr(audio_res, bts_num);
}


const char* parse_bytes(PyObject* args, PyObject *kwargs, unsigned long* bytes_number)
{
    static char* kwlist[] = {"encoded_bts", NULL};
    const char* bts = {0};
    Py_ssize_t bts_num = {0};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y#", kwlist, &bts, &bts_num)) return NULL;

    *bytes_number = (unsigned long) bts_num;

    return bts;
}


static PyObject*
g711_py_alaw_encode(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_encode(self, args, kwargs, g711_alaw_encode);
}


static PyObject*
g711_py_ulaw_encode(PyObject* self, PyObject* args, PyObject *kwargs)
{
    return g711_py_encode(self, args, kwargs, g711_ulaw_encode);
}


PyObject* g711_py_encode(PyObject* self, PyObject* args, PyObject *kwargs, EncodeFunction encode_fn)
{
    PyObject* arr = parse_array(args, kwargs);
    if (!arr) return NULL;

    float* arr_data = PyArray_DATA((PyArrayObject *)arr);
    unsigned long arr_len = PyArray_SIZE((PyArrayObject *)arr);

    char* bts = g711_encode(arr_data, arr_len, encode_fn);
    Py_DECREF(arr);

    if (!bts) return NULL;

    PyObject* res = Py_BuildValue("y#", bts, arr_len);

    free(bts);

    return res;
}


PyObject* parse_array(PyObject* args, PyObject *kwargs)
{
    PyObject* arr_obj = {0};

    static char *kwlist[] = {"audio_arr", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &arr_obj)) return NULL;
    return PyArray_FROM_OTF(arr_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
}


static PyMethodDef g711Methods[] = {
    {"load_alaw", (PyCFunction) g711_py_alaw_load, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified A-Law encoded audio file."},
    {"load_ulaw", (PyCFunction) g711_py_ulaw_load, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified u-Law encoded audio file."},
    {"save_alaw", (PyCFunction) g711_py_alaw_save, METH_VARARGS | METH_KEYWORDS, "Encode to A-Law and save the specified audio array."},
    {"save_ulaw", (PyCFunction) g711_py_ulaw_save, METH_VARARGS | METH_KEYWORDS, "Encode to u-Law and save the specified audio array."},
    {"decode_alaw", (PyCFunction) g711_py_alaw_decode, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified A-Law encoded audio file."},
    {"decode_ulaw", (PyCFunction) g711_py_ulaw_decode, METH_VARARGS | METH_KEYWORDS, "Load and decode the specified u-Law encoded audio file."},
    {"encode_alaw", (PyCFunction) g711_py_alaw_encode, METH_VARARGS | METH_KEYWORDS, "Encode to A-Law and save the specified audio array."},
    {"encode_ulaw", (PyCFunction) g711_py_ulaw_encode, METH_VARARGS | METH_KEYWORDS, "Encode to u-Law and save the specified audio array."},
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
