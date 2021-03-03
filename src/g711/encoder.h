#pragma once

#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif

bool g711_alaw_encode(unsigned long num_samples, const float in_buf[], char out_buf[]);
bool g711_ulaw_encode(unsigned long num_samples, const float in_buf[], char out_buf[]);

#ifdef __cplusplus
}
#endif
