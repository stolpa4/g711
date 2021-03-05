#pragma once

#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*DecodeFunction)(unsigned long, const char*, float*);


bool g711_alaw_decode(unsigned long num_samples, const char in_buf[], float out_buf[]);
bool g711_ulaw_decode(unsigned long num_samples, const char in_buf[], float out_buf[]);

#ifdef __cplusplus
}
#endif
