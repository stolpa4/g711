#pragma once


#ifdef __cplusplus
extern "C" {
#endif

float* g711_alaw_load(const char file_path[], unsigned long* samples_num);
float* g711_ulaw_load(const char file_path[], unsigned long* samples_num);

#ifdef __cplusplus
}
#endif
