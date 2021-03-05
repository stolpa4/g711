#pragma once


#include "g711/decoder.h"
#include "g711/encoder.h"


#ifdef __cplusplus
extern "C" {
#endif

float* g711_decode(const char bts[], unsigned long bts_num, DecodeFunction decoder);
char* g711_encode(const float audio_arr[], unsigned long samples_num, EncodeFunction encoder);

#ifdef __cplusplus
}
#endif
