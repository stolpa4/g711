#include "decoding_helpers.h"

#include "utils.h"


static inline short process_alaw_sample(short sample);
static inline short extract_alaw_mantissa(short preprocessed_sample);
static inline short invert_if_sample_negative(short sample, short mantissa);

static inline short process_ulaw_sample(short sample);
static inline short extract_sign(short sample);
static inline void extract_mantissa_exponent_step(short sample, short* mantissa,
                                                  short* exponent, short* lsb_position);
static inline short compute_ulaw_sample(short sign, short mantissa, short exponent, short lsb_position);


void g711_alaw_expand(unsigned long samples_num, const short buf_in[], float buf_out[])
{
    for (unsigned long sample_num = 0; sample_num < samples_num; sample_num++) {
        buf_out[sample_num] = g711_s16_to_float(process_alaw_sample(buf_in[sample_num]));
    }
}


short process_alaw_sample(short sample)
{
    /* re-toggle toggled bits */
    short preprocessed_sample = sample ^ 0x0055;

    /* remove sign bit */
    preprocessed_sample &= 0x007F;

    short mantissa = extract_alaw_mantissa(preprocessed_sample);

    return invert_if_sample_negative(sample, mantissa);
}


short extract_alaw_mantissa(short preprocessed_sample)
{
    /* extract exponent */
    short exponent = preprocessed_sample >> 4;

    /* get the mantissa */
    short mantissa = preprocessed_sample & 0x000F;

    /* add leading '1', if exponent > 0 */
    if (exponent > 0) mantissa += 0x0001 << 4;

    /* left-justify the mantissa */
    mantissa <<= 4;

    /* add 1/2 quantization step */
    mantissa += 0x0001 << 3;

    /* shift the mantissa left according to the exponent */
    if (exponent > 1) mantissa <<= exponent - 1;

    return mantissa;
}


short invert_if_sample_negative(short sample, short mantissa)
{
    return sample > 127 ? mantissa : -mantissa;
}


void g711_ulaw_expand(unsigned long samples_num, const short buf_in[], float buf_out[])
{
    for (unsigned long sample_num = 0; sample_num < samples_num; sample_num++) {
        buf_out[sample_num] = g711_s16_to_float(process_ulaw_sample(buf_in[sample_num]));
    }
}


short process_ulaw_sample(short sample)
{
    short sign = extract_sign(sample);

    short mantissa = {0};
    short exponent = {0};
    short lsb_position = {0};
    extract_mantissa_exponent_step(sample, &mantissa, &exponent, &lsb_position);

    return compute_ulaw_sample(sign, mantissa, exponent, lsb_position);
}


short extract_sign(short sample)
{
    /* sign-bit = 1 for positive values */
    return sample < 0x0080 ? -1 : 1;
}


void extract_mantissa_exponent_step(short sample, short* mantissa,
                                    short* exponent, short* lsb_position)
{
    /* 1's complement of input value */
    short sample_complement = ~sample;

    /* extract exponent */
    *exponent = (sample_complement >> 4) & 0x0007;

    /* extract mantissa */
    *mantissa = sample_complement & 0x000F;

    /* compute segment number */
    short segment = *exponent + 1;
    *lsb_position = 4 << segment;
}


short compute_ulaw_sample(short sign, short mantissa,
                          short exponent, short lsb_position)
{
    /* shift the mantissa left */
    short sample = lsb_position * mantissa;

    /* '1', preceding the mantissa */
    sample += 0x0080 << exponent;

    /* 1/2 quantization step */
    sample += lsb_position / 2;
    sample -= 4 * 33;

    return sign * sample;
}
