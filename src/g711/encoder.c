#include "encoder.h"

#include <stdlib.h>

#include "encoding_helpers.h"


typedef void (*CompressFunction)(unsigned long, float const*, short*);


static inline bool encode(unsigned long num_samples, const float in_buf[], char out_buf[], CompressFunction encoder);


bool g711_alaw_encode(unsigned long num_samples, const float in_buf[], char out_buf[])
{
    return encode(num_samples, in_buf, out_buf, g711_alaw_compress);
}


bool g711_ulaw_encode(unsigned long num_samples, const float in_buf[], char out_buf[])
{
    return encode(num_samples, in_buf, out_buf, g711_ulaw_compress);
}


bool encode(unsigned long num_samples, const float in_buf[], char out_buf[], CompressFunction encoder)
{
    short* tmp_buf = calloc(num_samples, sizeof(short));

    if (!tmp_buf) return false;

    encoder(num_samples, in_buf, tmp_buf);

    for (unsigned long i = 0; i < num_samples; ++i) {
        out_buf[i] = (char) (tmp_buf[i] & 0x00FF);
    }

    free(tmp_buf);

    return true;    
}
