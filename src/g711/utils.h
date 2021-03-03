#pragma once

#include <stdbool.h>
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

/* Sample convertors */
float g711_s16_to_float(short sample);
short g711_float_to_s16(float sample);

/* Filesystem */
bool g711_validate_path(const char file_path[]);
FILE* g711_open_file_for_read(const char file_path[]);
unsigned long g711_get_file_bts_num(FILE* file);
char* g711_read_file(FILE* file, unsigned long bts_num);
FILE* g711_open_file_for_write(const char file_path[]);
bool g711_write_file(FILE* file, const char* bts, unsigned long bts_num);
void g711_close_file(FILE* file);

#ifdef __cplusplus
}
#endif
