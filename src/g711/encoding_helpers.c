#include "encoding_helpers.h"

#include "utils.h"


static inline short process_alaw_sample(short sample);
static inline short handle_alaw_negatives(short sample);
static inline short process_exponent(short compressed);

static inline short process_ulaw_sample(short sample);
static inline short compute_abs_val(short sample);
static inline short compute_segment_no(short abs_val);
static inline void compute_nibbles(short segment, short abs_val, short* high, short* low);
static inline short add_sign_bit(short sample, short compressed);


void g711_alaw_compress(unsigned long samples_num, const float buf_in[], short buf_out[])
{
    for (unsigned long sample_num = 0; sample_num < samples_num; sample_num++) {
        buf_out[sample_num] = process_alaw_sample(g711_float_to_s16(buf_in[sample_num]));
    }
}


short process_alaw_sample(short sample)
{
    short compressed = handle_alaw_negatives(sample);

    /* if exponent > 0 */
    if (compressed > 15) compressed = process_exponent(compressed);

    /* add sign bit */
    if (sample >= 0) compressed |= 0x0080;

    /* toggle even bits */
    compressed ^= 0x0055;

    return compressed;
}


short handle_alaw_negatives(short sample)
{
    /* 1's complement for negative values */
    if (sample < 0) sample = ~sample;

    /* 0 <= sample < 2048 */
    return sample >> 4;
}


short process_exponent(short compressed)
{
    short exponent = 1;

    /* Find mantissa and exponent */
    while (compressed > 16 + 15) {
        compressed >>= 1;
        exponent++;
    }

    /* Remove leading '1' */
    compressed -= 0x0001 << 4;

    /* Compute the encoded value */
    compressed += exponent << 4;

    return compressed;
}


void g711_ulaw_compress(unsigned long samples_num, const float buf_in[], short buf_out[])
{
    for (unsigned long sample_num = 0; sample_num < samples_num; sample_num++) {
        buf_out[sample_num] = process_ulaw_sample(g711_float_to_s16(buf_in[sample_num]));
    }
}


short process_ulaw_sample(short sample)
{
    short abs_val = compute_abs_val(sample);
    short segment = compute_segment_no(abs_val);

    short high_nibble = {0};
    short low_nibble = {0};
    compute_nibbles(segment, abs_val, &high_nibble, &low_nibble);

    short compressed = (high_nibble << 4) | low_nibble;

    return add_sign_bit(sample, compressed);
}


short compute_abs_val(short sample)
{
    /* compute 1's complement in case of negative sample */
    if (sample < 0) sample = ~sample;

    sample >>= 2;

    /* 33 is the difference value between the thresholds for A-law and u-law */
    short abs_val = sample + 33;

    /* Abs val limit: 8192 */
    if (abs_val > 0x1FFF) abs_val = 0x1FFF;

    return abs_val;
}


short compute_segment_no(short abs_val)
{
    short segment = 1;

    short seg_cnt = abs_val >> 6;

    while (seg_cnt != 0) {
        segment++;
        seg_cnt >>= 1;
    }

    return segment;
}


void compute_nibbles(short segment, short abs_val, short* high, short* low)
{
    /* Mounting the high-nibble of the log-PCM sample */
    *high = 0x0008 - segment;

    /* Mounting the low-nibble of the log PCM sample */
    /* right shift of mantissa and */
    *low = abs_val >> segment;

    /* masking away leading '1' */
    *low &= 0x000F;

    *low = 0x000F - *low;
}


short add_sign_bit(short sample, short compressed)
{
    if (sample >= 0) compressed |= 0x0080;
    return compressed;
}
