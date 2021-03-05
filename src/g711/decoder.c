#include "decoder.h"

#include <stdlib.h>

#include "decoding_helpers.h"


typedef void (*ExpandFunction)(unsigned long, short const*, float*);


static inline bool decode(unsigned long num_samples, const char in_buf[], float out_buf[], ExpandFunction decoder);


bool g711_alaw_decode(unsigned long num_samples, const char in_buf[], float out_buf[])
{
    return decode(num_samples, in_buf, out_buf, g711_alaw_expand);
}


bool g711_ulaw_decode(unsigned long num_samples, const char in_buf[], float out_buf[])
{
    return decode(num_samples, in_buf, out_buf, g711_ulaw_expand);
}


bool decode(unsigned long num_samples, const char in_buf[], float out_buf[], ExpandFunction decoder)
{
    short* tmp_buf = malloc(num_samples * sizeof(short));

    if (!tmp_buf) return false;

    for (unsigned long i = 0; i < num_samples; ++i) {
        tmp_buf[i] = (short) (in_buf[i] & 0x00FF);
    }

    decoder(num_samples, tmp_buf, out_buf);

    free(tmp_buf);

    return true;
}
