#pragma once


#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif

bool g711_alaw_save(const char file_path[], const float audio_arr[], unsigned long samples_num);
bool g711_ulaw_save(const char file_path[], const float audio_arr[], unsigned long samples_num);

#ifdef __cplusplus
}
#endif
