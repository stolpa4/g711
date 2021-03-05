#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "g711module_saver.h"

#include <stddef.h>
#include <stdio.h>

#include "g711module_coder.h"
#include "g711/utils.h"


static inline bool save(const char file_path[], const float audio_arr[], unsigned long samples_num, EncodeFunction encoder);
static inline bool validate_args(const char file_path[], unsigned long samples_num);
static inline bool write_bts(const char file_path[], const char bts[], unsigned long bts_num);


bool g711_alaw_save(const char file_path[], const float audio_arr[], unsigned long samples_num)
{
    return save(file_path, audio_arr, samples_num, g711_alaw_encode);
}


bool g711_ulaw_save(const char file_path[], const float audio_arr[], unsigned long samples_num)
{
    return save(file_path, audio_arr, samples_num, g711_ulaw_encode);
}


bool save(const char file_path[], const float audio_arr[], unsigned long samples_num, EncodeFunction encoder)
{
    if (!validate_args(file_path, samples_num)) return false;

    /* 1 byte == 1 sample */
    char* audio_bts = g711_encode(audio_arr, samples_num, encoder);
    if (!audio_bts) return false;

    bool status_ok = write_bts(file_path, audio_bts, samples_num);

    free(audio_bts);

    return status_ok;
}


bool validate_args(const char file_path[], unsigned long samples_num)
{
    if (!g711_validate_path(file_path)) {
        PyErr_SetString(PyExc_ValueError, "The function accepts only non-empty string or path-like objects.");
        return false;
    }

    if (!samples_num) {
        PyErr_SetString(PyExc_ValueError, "Unable to save an empty array.");
        return false;
    }

    return true;
}


bool write_bts(const char file_path[], const char bts[], unsigned long bts_num)
{
    FILE* file = g711_open_file_for_write(file_path);
    if (!file) {
        PyErr_Format(PyExc_FileNotFoundError, "Unable to open file: %s. Check if the parent dir does exist and has proper permissions.", file_path);
        return false;
    }

    bool status_ok = g711_write_file(file, bts, bts_num);

    g711_close_file(file);

    if (!status_ok) PyErr_Format(PyExc_IOError, "Failed writing to file: %s.", file_path);

    return status_ok;
}
