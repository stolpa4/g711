#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void g711_alaw_compress(unsigned long samples_num, const float buf_in[], short buf_out[]);

void g711_ulaw_compress(unsigned long samples_num, const float buf_in[], short buf_out[]);

#ifdef __cplusplus
}
#endif
