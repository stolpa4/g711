#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void g711_alaw_expand(unsigned long samples_num, short const buf_in[], float buf_out[]);

void g711_ulaw_expand(unsigned long samples_num, short const buf_in[], float buf_out[]);

#ifdef __cplusplus
}
#endif
