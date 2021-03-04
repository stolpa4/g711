#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "g711module_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "g711/decoder.h"
#include "g711/utils.h"


typedef float* (*DecodeFunction)(unsigned long, const char*, float*);


static inline float* load(const char file_path[], unsigned long* samples_num, DecodeFunction decoder);
static inline char* read_bts(const char file_path[], unsigned long* bts_num);
static inline float* decode_bts(const char bts[], unsigned long bts_num, DecodeFunction decoder);


float* g711_alaw_load(const char file_path[], unsigned long* samples_num)
{
    return load(file_path, samples_num, g711_alaw_decode);
}


float* g711_ulaw_load(const char file_path[], unsigned long* samples_num)
{
    return load(file_path, samples_num, g711_ulaw_decode);
}


float* load(const char file_path[], unsigned long* samples_num, DecodeFunction decoder)
{
    /* 1 byte == 1 sample */
    char* audio_bts = read_bts(file_path, samples_num);

    if (!audio_bts) return NULL;

    float* audio_res = decode_bts(audio_bts, *samples_num, decoder);

    free(audio_bts);

    return audio_res;
}


char* read_bts(const char file_path[], unsigned long* bts_num)
{
    if (!g711_validate_path(file_path)) {
        PyErr_SetString(PyExc_ValueError, "The function accepts only non-empty string or path-like objects.");
        return NULL;
    }

    FILE* file = g711_open_file_for_read(file_path);
    if (!file) {
        PyErr_Format(PyExc_FileNotFoundError, "Unable to open file: %s. Check if it exists and has proper permissions.", file_path);
        return NULL;
    }

    *bts_num = g711_get_file_bts_num(file);
    if (!*bts_num) {
        PyErr_Format(PyExc_IOError, "Unable to determine the size of file: %s.", file_path);
        return NULL;
    }

    char* bts = g711_read_file(file, *bts_num);
    
    if (!bts) {
        PyErr_Format(PyExc_IOError, "Error reading the file: %s.", file_path);
        return NULL;
    }

    g711_close_file(file);

    return bts;
}


float* decode_bts(const char bts[], unsigned long bts_num, DecodeFunction decoder)
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
