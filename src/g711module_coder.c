#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdlib.h>

#include "g711module_coder.h"


float* g711_decode(const char bts[], unsigned long bts_num, DecodeFunction decoder)
{
    float* audio_res = malloc(sizeof(float) * bts_num);

    if (!audio_res) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    bool status_ok = decoder(bts_num, bts, audio_res);

    if (!status_ok) {
        free(audio_res);
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    return audio_res;
}


char* g711_encode(const float audio_arr[], unsigned long samples_num, EncodeFunction encoder)
{
    char* audio_bts = malloc(samples_num);

    if (!audio_bts) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    bool status_ok = encoder(samples_num, audio_arr, audio_bts);

    if (!status_ok) {
        free(audio_bts);
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    return audio_bts;
}
